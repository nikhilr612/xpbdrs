//! Tetrahedral mesh implementation.

use bitvec::vec::BitVec;
use raylib::prelude::*;
use std::collections::HashMap;
use std::io::Write;
use tracing::{debug, error, info};

use super::common::{Result, TetrahedronId, Triangle, Vertex, dedup_with_warning};
use super::tgimport::TetgenParser;
use crate::constraint::Constraint;
use crate::mesh::{Edge, Tetrahedron};
use crate::xpbd::{ConstraintSet, XpbdState};

/// State for tracking torn faces in a tetrahedral mesh.
/// This is separate from XpbdState as it contains mesh-topology-specific information.
/// Also contains precomputed adjacency data for efficient tearing cascades.
pub struct MeshTearState {
    /// Boolean vector indicating torn faces by index.
    torn_faces: BitVec,
    /// Map from constraint edge index to list of face indices that reference it.
    edge_to_faces: HashMap<usize, Vec<usize>>,
    /// Map from vertex ID to list of constraint edge indices incident to that vertex.
    vertex_to_edges: HashMap<u32, Vec<usize>>,
}

impl MeshTearState {
    /// Create a new tear state for a mesh with the given number of faces.
    /// Use `with_adjacency` for efficient tearing cascade operations.
    #[must_use]
    pub fn new(n_faces: usize) -> Self {
        Self {
            torn_faces: BitVec::repeat(false, n_faces),
            edge_to_faces: HashMap::new(),
            vertex_to_edges: HashMap::new(),
        }
    }
    
    /// Create a new tear state with precomputed adjacency data for efficient tearing.
    #[must_use]
    pub fn with_adjacency(faces: &[Triangle], edges: &[Edge]) -> Self {
        let n_faces = faces.len();
        
        // Build edge→face adjacency map from face.edges (authoritative source)
        let mut edge_to_faces: HashMap<usize, Vec<usize>> = HashMap::new();
        for (face_idx, face) in faces.iter().enumerate() {
            for edge_id in face.edges.iter().filter_map(|e| e.as_ref()) {
                edge_to_faces.entry(edge_id.0 as usize).or_default().push(face_idx);
            }
        }
        
        // Build vertex→edge adjacency map
        let mut vertex_to_edges: HashMap<u32, Vec<usize>> = HashMap::new();
        for (edge_idx, edge) in edges.iter().enumerate() {
            vertex_to_edges.entry(edge.0.0).or_default().push(edge_idx);
            vertex_to_edges.entry(edge.1.0).or_default().push(edge_idx);
        }
        
        Self {
            torn_faces: BitVec::repeat(false, n_faces),
            edge_to_faces,
            vertex_to_edges,
        }
    }
    
    /// Check if a face at given index is torn.
    #[must_use]
    pub fn face_torn(&self, index: usize) -> bool {
        self.torn_faces
            .as_bitslice()
            .get(index)
            .is_some_and(|b| *b)
    }
    
    /// Mark a face as torn.
    pub fn tear_face(&mut self, index: usize) {
        if index < self.torn_faces.len() {
            self.torn_faces.set(index, true);
        }
    }
    
    /// Reset all faces to non-torn state.
    pub fn reset(&mut self) {
        self.torn_faces.fill(false);
    }
    
    /// Get faces that reference the given constraint edge index.
    #[must_use]
    pub fn faces_for_edge(&self, edge_idx: usize) -> &[usize] {
        self.edge_to_faces.get(&edge_idx).map_or(&[], |v| v.as_slice())
    }
    
    /// Get constraint edge indices incident to the given vertex.
    #[must_use]
    pub fn edges_for_vertex(&self, vertex_id: u32) -> &[usize] {
        self.vertex_to_edges.get(&vertex_id).map_or(&[], |v| v.as_slice())
    }
}

/// Values computed from tetrahedral constraints.
pub struct TetConstraintValues {
    /// Edge lengths for distance constraints.
    pub lengths: Vec<f32>,
    /// Tetrahedron volumes for volume constraints.
    pub volumes: Vec<f32>,
}

/// Struct to contain constraint data for tetrahedral meshes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TetConstraints {
    /// Edge constraints for distance preservation.
    pub edges: Vec<Edge>,
    /// Tetrahedral constraints for volume preservation.
    pub tetrahedra: Vec<Tetrahedron>,
}

impl ConstraintSet<Vec<Vertex>, TetConstraintValues> for TetConstraints {
    fn evaluate(&self, on: &Vec<Vertex>) -> TetConstraintValues {
        let lengths = self.edges.iter().map(|e| e.value(on)).collect();
        let volumes = self.tetrahedra.iter().map(|t| t.value(on)).collect();
        TetConstraintValues { lengths, volumes }
    }

    fn solve(
        &self,
        processor: crate::xpbd::ConstraintProcessor<Vec<Vertex>>,
        params: &crate::xpbd::XpbdParams,
        reference: &TetConstraintValues,
    ) {
        let dt_squared = params.time_substep * params.time_substep;
        let _ = processor
            .process(
                self.edges.iter().zip(reference.lengths.iter().copied()),
                params.l_threshold_length,
                params.length_compliance / dt_squared,
                dt_squared,
                params.shuffle_buffer_size,
            )
            .process(
                self.tetrahedra
                    .iter()
                    .zip(reference.volumes.iter().map(|v| v * params.p_volume)),
                params.l_threshold_volume,
                params.volume_compliance / dt_squared,
                dt_squared,
                params.shuffle_buffer_size,
            );
    }
}

/// Struct to contain data of a delaunay tetrahedralized mesh.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Tetrahedral {
    /// Vertices of the tetrahedral mesh.
    pub vertices: Vec<Vertex>,
    /// Constraints for physics simulation.
    pub constraints: TetConstraints,
    /// Triangular faces of the mesh.
    pub faces: Vec<Triangle>,
}

impl Tetrahedral {
    /// Get the corner points of a tetrahedron by its ID.
    ///
    /// # Returns
    /// `None` if the tetrahedron ID is invalid or references invalid vertices.
    #[must_use]
    pub fn corners(&self, id: TetrahedronId) -> Option<[Vector3; 4]> {
        let tet = self.constraints.tetrahedra.get(id.0 as usize)?;
        Some([
            self.vertices[tet.indices[0]].position,
            self.vertices[tet.indices[1]].position,
            self.vertices[tet.indices[2]].position,
            self.vertices[tet.indices[3]].position,
        ])
    }

    /// Load tetrahedral mesh from tetgen files.
    ///
    /// # Errors
    /// Returns an error if files cannot be read or parsed.
    #[tracing::instrument]
    pub fn from_files(prefix: &str) -> Result<Self> {
        // Deduplication is now streamlined using custom Hash and Eq implementations
        let vertices = TetgenParser::load_vertices(prefix)?;
        let edges = dedup_with_warning(TetgenParser::load_edges(prefix)?, "edge");
        let face_triangles = TetgenParser::load_face_vertices(prefix)?;
        let (edges, faces) = TetgenParser::build_faces_with_edges(edges, face_triangles);
        let tetrahedra = dedup_with_warning(TetgenParser::load_tetrahedra(prefix)?, "tetrahedron");

        let result = Self {
            vertices,
            constraints: TetConstraints { edges, tetrahedra },
            faces,
        };

        info!(
            vertices = result.vertices.len(),
            edges = result.constraints.edges.len(),
            faces = result.faces.len(),
            tetrahedra = result.constraints.tetrahedra.len(),
            "Mesh loaded from tetgen files"
        );

        Ok(result)
    }

    /// Load tetrahedral mesh from bincode file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or deserialized.
    #[tracing::instrument]
    pub fn from_bincode(filename: &str) -> Result<Self> {
        let data = std::fs::read(filename)?;
        debug!("Deserializing {} bytes", data.len());
        let mesh: Self = bincode::deserialize(&data)?;
        Ok(mesh)
    }

    /// Export mesh to bincode format.
    ///
    /// # Errors
    /// Returns an error if serialization fails or file cannot be written.
    #[tracing::instrument(skip(self))]
    pub fn export_to_bincode(&self, output_path: &str) -> Result<()> {
        info!("Serializing to binary format");
        let encoded = bincode::serialize(self)?;

        let mut file = std::fs::File::create(output_path)?;
        file.write_all(&encoded)?;

        info!(
            output_path,
            size_bytes = encoded.len(),
            "Successfully exported mesh"
        );

        // Verify deserialization works
        debug!("Verifying serialized data");
        let _: Self = bincode::deserialize(&encoded)?;
        debug!("Verification successful");

        Ok(())
    }

    /// Load mesh with automatic format detection.
    ///
    /// # Errors
    /// Returns an error if the file format is unsupported or loading fails.
    #[tracing::instrument]
    pub fn load_mesh(mesh_path: &str) -> Result<Self> {
        info!(mesh_path, "Attempting to load mesh");

        let mesh = if std::path::Path::new(mesh_path)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
        {
            debug!("Loading as bincode file");
            Self::from_bincode(mesh_path)
        } else {
            debug!("Loading as tetgen files");
            Self::from_files(mesh_path)
        };

        match &mesh {
            Ok(m) => {
                info!(
                    vertices = m.vertices.len(),
                    edges = m.constraints.edges.len(),
                    faces = m.faces.len(),
                    tetrahedra = m.constraints.tetrahedra.len(),
                    "Mesh loaded successfully"
                );
            }
            Err(e) => {
                error!(mesh_path, error = %e, "Failed to load mesh");
            }
        }

        mesh
    }

    /// Draw wireframe of the mesh. Only draws edges that are active and referenced by at least one non-torn face.
    pub fn draw_wireframe(&self, d3: &mut RaylibMode3D<RaylibDrawHandle>, xpbd_state: &XpbdState, tear_state: &MeshTearState, color: Color) {
        for (edge_idx, edge) in self.constraints.edges.iter().enumerate() {
            // Skip inactive (removed) edges
            if xpbd_state.constraint_inactive(edge_idx) {
                continue;
            }
            
            // Check if this edge is referenced by any non-torn face
            let has_active_face = tear_state.faces_for_edge(edge_idx)
                .iter()
                .any(|&face_idx| !tear_state.face_torn(face_idx));
            
            if !has_active_face {
                continue;
            }
            
            // Bounds-safe vertex access
            let v0_idx = (edge.0.0 as usize).saturating_sub(1);
            let v1_idx = (edge.1.0 as usize).saturating_sub(1);
            
            if let (Some(vert1), Some(vert2)) = (
                self.vertices.get(v0_idx),
                self.vertices.get(v1_idx),
            ) {
                d3.draw_line_3D(vert1.position, vert2.position, color);
            }
        }
    }

    /// Draw filled faces. Only draws faces that are not torn and have no inactive edges.
    pub fn draw_faces(
        &self,
        d3: &mut RaylibMode3D<RaylibDrawHandle>,
        xpbd_state: &XpbdState,
        tear_state: &MeshTearState,
        color: Color,
    ) {
        for (face_idx, face) in self.faces.iter().enumerate() {
            // Skip torn faces
            if tear_state.face_torn(face_idx) {
                continue;
            }
            
            // Also skip faces that have any inactive edge (belt-and-suspenders check)
            let has_inactive_edge = face.edges.iter()
                .filter_map(|e| e.as_ref())
                .any(|edge_id| xpbd_state.constraint_inactive(edge_id.0 as usize));
            
            if has_inactive_edge {
                continue;
            }
            
            // Bounds-safe vertex access
            let v0_idx = (face.verts[0].0 as usize).saturating_sub(1);
            let v1_idx = (face.verts[1].0 as usize).saturating_sub(1);
            let v2_idx = (face.verts[2].0 as usize).saturating_sub(1);
            
            let (Some(v0), Some(v1), Some(v2)) = (
                self.vertices.get(v0_idx),
                self.vertices.get(v1_idx),
                self.vertices.get(v2_idx),
            ) else {
                continue;
            };

            d3.draw_triangle3D(v0.position, v1.position, v2.position, color);
        }
    }
    /// Tear edges based on deformation thresholds and cascade removals.
    ///
    /// Logic:
    /// 1. Remove edges exceeding stretch/compression thresholds
    /// 2. Remove faces that have ANY removed edge (using adjacency map if available)
    /// 3. Remove edges where BOTH vertices only belong to torn faces
    /// 4. Repeat 2-3 until no changes
    /// 5. Freeze vertices with no remaining edges
    pub fn tear_edges(
        &self,
        xpbd_state: &mut XpbdState,
        tear_state: &mut MeshTearState,
        initial_values: &TetConstraintValues,
        stretch_threshold: f32,
        compression_threshold: f32,
    ) {
        // Step 1: Remove edges exceeding thresholds
        const MIN_LENGTH_EPSILON: f32 = 1e-8;
        
        for (edge_idx, edge) in self.constraints.edges.iter().enumerate() {
            if xpbd_state.constraint_inactive(edge_idx) {
                continue;
            }
            
            // Bounds-safe vertex access
            let v0_idx = (edge.0.0 as usize).saturating_sub(1);
            let v1_idx = (edge.1.0 as usize).saturating_sub(1);
            
            let (Some(v0), Some(v1)) = (self.vertices.get(v0_idx), self.vertices.get(v1_idx)) else {
                // Invalid vertex indices - deactivate this constraint
                xpbd_state.deactivate_constraint(edge_idx);
                continue;
            };
            
            let current_length = (v1.position - v0.position).length();
            let original_length = initial_values.lengths.get(edge_idx).copied().unwrap_or(0.0);
            
            // Guard against division by zero: skip edges with near-zero original length
            if original_length < MIN_LENGTH_EPSILON {
                continue;
            }
            
            let ratio = current_length / original_length;
            
            if ratio > stretch_threshold || ratio < compression_threshold {
                xpbd_state.deactivate_constraint(edge_idx);
            }
        }
        
        // Step 2: Tear all faces that have any inactive edge
        // This is a one-way cascade: edge deactivation → face tearing
        // We don't deactivate edges just because their faces are torn
        for (face_idx, face) in self.faces.iter().enumerate() {
            if tear_state.face_torn(face_idx) {
                continue;
            }
            
            let has_inactive_edge = face.edges.iter()
                .filter_map(|e| e.as_ref())
                .any(|edge_id| xpbd_state.constraint_inactive(edge_id.0 as usize));
            
            if has_inactive_edge {
                tear_state.tear_face(face_idx);
            }
        }
        
        // Step 3: Freeze vertices with no active edges
        for (vert_idx, _) in self.vertices.iter().enumerate() {
            let vert_id = (vert_idx + 1) as u32;
            
            let has_active_edge = tear_state.edges_for_vertex(vert_id)
                .iter()
                .any(|&edge_idx| !xpbd_state.constraint_inactive(edge_idx));
            
            if !has_active_edge {
                xpbd_state.dampen_vertex_velocity(vert_idx, 0.0);
            }
        }
        
        // Step 4: Detect disconnected fragments via flood fill from anchored vertices
        self.tear_disconnected_fragments(xpbd_state, tear_state);
    }
    
    /// Tear faces that are disconnected from the main mesh.
    /// Uses flood fill from "anchored" vertices (at ground level) to find connected component.
    fn tear_disconnected_fragments(
        &self,
        xpbd_state: &mut XpbdState,
        tear_state: &mut MeshTearState,
    ) {
        use std::collections::{HashSet, VecDeque};
        
        // Find anchored vertices (at or near ground level, or part of largest component)
        let mut anchored_verts: HashSet<u32> = HashSet::new();
        for (vert_idx, vert) in self.vertices.iter().enumerate() {
            if vert.position.y <= 0.05 {
                anchored_verts.insert((vert_idx + 1) as u32);
            }
        }
        
        // If no ground-level vertices, use the vertex with most active edges as anchor
        if anchored_verts.is_empty() {
            let mut best_vert = 1u32;
            let mut best_count = 0;
            for (vert_idx, _) in self.vertices.iter().enumerate() {
                let vert_id = (vert_idx + 1) as u32;
                let count = self.constraints.edges.iter().enumerate()
                    .filter(|(edge_idx, edge)| {
                        !xpbd_state.constraint_inactive(*edge_idx) && 
                        (edge.0.0 == vert_id || edge.1.0 == vert_id)
                    })
                    .count();
                if count > best_count {
                    best_count = count;
                    best_vert = vert_id;
                }
            }
            anchored_verts.insert(best_vert);
        }
        
        // Flood fill to find all vertices connected to anchored vertices via active edges
        let mut connected_verts: HashSet<u32> = HashSet::new();
        let mut queue: VecDeque<u32> = anchored_verts.iter().copied().collect();
        
        while let Some(vert_id) = queue.pop_front() {
            if connected_verts.contains(&vert_id) {
                continue;
            }
            connected_verts.insert(vert_id);
            
            // Find neighbors via active edges
            for (edge_idx, edge) in self.constraints.edges.iter().enumerate() {
                if xpbd_state.constraint_inactive(edge_idx) {
                    continue;
                }
                
                let neighbor = if edge.0.0 == vert_id {
                    Some(edge.1.0)
                } else if edge.1.0 == vert_id {
                    Some(edge.0.0)
                } else {
                    None
                };
                
                if let Some(n) = neighbor {
                    if !connected_verts.contains(&n) {
                        queue.push_back(n);
                    }
                }
            }
        }
        
        // Tear faces where ANY vertex is not connected
        for (face_idx, face) in self.faces.iter().enumerate() {
            if tear_state.face_torn(face_idx) {
                continue;
            }
            
            let all_connected = face.verts.iter()
                .all(|v| connected_verts.contains(&v.0));
            
            if !all_connected {
                tear_state.tear_face(face_idx);
            }
        }
        
        // Deactivate edges where either vertex is not connected
        for (edge_idx, edge) in self.constraints.edges.iter().enumerate() {
            if xpbd_state.constraint_inactive(edge_idx) {
                continue;
            }
            
            let both_connected = connected_verts.contains(&edge.0.0) && 
                                 connected_verts.contains(&edge.1.0);
            
            if !both_connected {
                xpbd_state.deactivate_constraint(edge_idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Edge, Spatial, Tetrahedron, VertexId, common::EdgeId};
    use crate::xpbd::XpbdState;
    use std::fs;

    fn create_test_files(prefix: &str) {
        fs::write(
            format!("{prefix}.node"),
            "2 3 0 0\n1 0.0 0.0 0.0\n2 1.0 1.0 1.0\n",
        )
        .unwrap();
        fs::write(format!("{prefix}.edge"), "1 0\n1 1 2\n").unwrap();
        fs::write(format!("{prefix}.face"), "1 0\n1 1 2 1\n").unwrap();
        fs::write(format!("{prefix}.ele"), "1 4 0\n1 1 2 1 2\n").unwrap();
    }

    #[test]
    fn test_parse() {
        let prefix = "test";
        create_test_files(prefix);

        let mesh = Tetrahedral::from_files(prefix).unwrap();
        assert_eq!(mesh.vertices.len(), 2);
        assert_eq!(mesh.constraints.edges.len(), 1);
        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.constraints.tetrahedra.len(), 1);

        // Cleanup
        for ext in &["node", "edge", "face", "ele"] {
            let _ = fs::remove_file(format!("{prefix}.{ext}"));
        }
    }

    #[test]
    fn test_translation() {
        let prefix = "test_translate";
        create_test_files(prefix);

        let mut mesh = Tetrahedral::from_files(prefix).unwrap();
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let original_positions: Vec<_> = mesh.vertices.iter().map(|v| v.position).collect();

        mesh.vertices.translate(translation);

        for (i, vertex) in mesh.vertices.iter().enumerate() {
            let expected = original_positions[i] + translation;
            assert!((vertex.position.x - expected.x).abs() < f32::EPSILON);
            assert!((vertex.position.y - expected.y).abs() < f32::EPSILON);
            assert!((vertex.position.z - expected.z).abs() < f32::EPSILON);
        }

        // Cleanup
        for ext in &["node", "edge", "face", "ele"] {
            let _ = fs::remove_file(format!("{prefix}.{ext}"));
        }
    }

    #[test]
    fn test_bounding_box() {
        let prefix = "test_bbox";
        create_test_files(prefix);

        let mesh = Tetrahedral::from_files(prefix).unwrap();
        let (min, max) = mesh.vertices.bounding_box();

        // Based on our test data: vertices at (0,0,0) and (1,1,1)
        assert_eq!(min, Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(max, Vector3::new(1.0, 1.0, 1.0));

        // Cleanup
        for ext in &["node", "edge", "face", "ele"] {
            let _ = fs::remove_file(format!("{prefix}.{ext}"));
        }
    }
    
    // Helper to create a simple mesh for tearing tests
    fn create_tear_test_mesh() -> (Tetrahedral, TetConstraintValues) {
        // Create a simple mesh with 4 vertices forming a tetrahedron
        let vertices = vec![
            Vertex { position: Vector3::new(0.0, 0.0, 0.0), inv_mass: 1.0 },
            Vertex { position: Vector3::new(1.0, 0.0, 0.0), inv_mass: 1.0 },
            Vertex { position: Vector3::new(0.5, 1.0, 0.0), inv_mass: 1.0 },
            Vertex { position: Vector3::new(0.5, 0.5, 1.0), inv_mass: 1.0 },
        ];
        
        // Edges (using 1-based indexing as per tetgen format)
        let edges = vec![
            Edge(VertexId(1), VertexId(2)),
            Edge(VertexId(2), VertexId(3)),
            Edge(VertexId(3), VertexId(1)),
            Edge(VertexId(1), VertexId(4)),
            Edge(VertexId(2), VertexId(4)),
            Edge(VertexId(3), VertexId(4)),
        ];
        
        // Faces
        let faces = vec![
            Triangle { 
                verts: [VertexId(1), VertexId(2), VertexId(3)],
                edges: [Some(EdgeId(0)), Some(EdgeId(1)), Some(EdgeId(2))],
            },
            Triangle { 
                verts: [VertexId(1), VertexId(2), VertexId(4)],
                edges: [Some(EdgeId(0)), Some(EdgeId(4)), Some(EdgeId(3))],
            },
            Triangle { 
                verts: [VertexId(2), VertexId(3), VertexId(4)],
                edges: [Some(EdgeId(1)), Some(EdgeId(5)), Some(EdgeId(4))],
            },
            Triangle { 
                verts: [VertexId(3), VertexId(1), VertexId(4)],
                edges: [Some(EdgeId(2)), Some(EdgeId(3)), Some(EdgeId(5))],
            },
        ];
        
        let mesh = Tetrahedral {
            vertices,
            constraints: TetConstraints {
                edges,
                tetrahedra: vec![Tetrahedron { indices: [VertexId(1), VertexId(2), VertexId(3), VertexId(4)] }],
            },
            faces,
        };
        
        let initial_values = mesh.constraints.evaluate(&mesh.vertices);
        (mesh, initial_values)
    }
    
    #[test]
    fn test_tear_edges_single_edge_removal() {
        let (mut mesh, initial_values) = create_tear_test_mesh();
        let mut xpbd_state = XpbdState::new(
            mesh.vertices.len(),
            mesh.constraints.edges.len() + mesh.constraints.tetrahedra.len(),
        );
        let mut tear_state = MeshTearState::with_adjacency(&mesh.faces, &mesh.constraints.edges);
        
        // Stretch edge 0 (between vertices 1 and 2) beyond threshold
        mesh.vertices[1].position = Vector3::new(3.0, 0.0, 0.0); // Move far away
        
        // Threshold at 150% should trigger tearing
        mesh.tear_edges(&mut xpbd_state, &mut tear_state, &initial_values, 1.5, 0.5);
        
        // Edge 0 should be deactivated (stretched beyond 150%)
        assert!(xpbd_state.constraint_inactive(0), "Stretched edge should be deactivated");
    }
    
    #[test]
    fn test_tear_edges_cascade_face_removal() {
        let (mesh, initial_values) = create_tear_test_mesh();
        let mut xpbd_state = XpbdState::new(
            mesh.vertices.len(),
            mesh.constraints.edges.len() + mesh.constraints.tetrahedra.len(),
        );
        let mut tear_state = MeshTearState::with_adjacency(&mesh.faces, &mesh.constraints.edges);
        
        // Manually deactivate edge 0 (shared by faces 0 and 1)
        xpbd_state.deactivate_constraint(0);
        
        // Run tear cascade
        mesh.tear_edges(&mut xpbd_state, &mut tear_state, &initial_values, 2.0, 0.1);
        
        // Faces containing edge 0 should be torn
        assert!(tear_state.face_torn(0), "Face 0 should be torn (contains deactivated edge)");
        assert!(tear_state.face_torn(1), "Face 1 should be torn (contains deactivated edge)");
    }
    
    #[test]
    fn test_tear_edges_compression_threshold() {
        let (mut mesh, initial_values) = create_tear_test_mesh();
        let mut xpbd_state = XpbdState::new(
            mesh.vertices.len(),
            mesh.constraints.edges.len() + mesh.constraints.tetrahedra.len(),
        );
        let mut tear_state = MeshTearState::with_adjacency(&mesh.faces, &mesh.constraints.edges);
        
        // Compress edge 0 (between vertices 1 and 2) below threshold
        mesh.vertices[1].position = Vector3::new(0.1, 0.0, 0.0); // Move very close
        
        // Compression threshold at 50% should trigger tearing
        mesh.tear_edges(&mut xpbd_state, &mut tear_state, &initial_values, 2.0, 0.5);
        
        // Edge 0 should be deactivated (compressed below 50%)
        assert!(xpbd_state.constraint_inactive(0), "Compressed edge should be deactivated");
    }
    
    #[test]
    fn test_tear_edges_isolated_vertex_freezing() {
        let (mesh, initial_values) = create_tear_test_mesh();
        let mut xpbd_state = XpbdState::new(
            mesh.vertices.len(),
            mesh.constraints.edges.len() + mesh.constraints.tetrahedra.len(),
        );
        let mut tear_state = MeshTearState::with_adjacency(&mesh.faces, &mesh.constraints.edges);
        
        // Set some velocity
        xpbd_state.velocities_mut()[0] = Vector3::new(1.0, 1.0, 1.0);
        
        // Deactivate all edges connected to vertex 1 (indices 0, 2, 3)
        xpbd_state.deactivate_constraint(0); // Edge 1-2
        xpbd_state.deactivate_constraint(2); // Edge 3-1
        xpbd_state.deactivate_constraint(3); // Edge 1-4
        
        // Run tear cascade
        mesh.tear_edges(&mut xpbd_state, &mut tear_state, &initial_values, 2.0, 0.1);
        
        // Vertex 0 (index 0) should have its velocity dampened to zero
        let vel = xpbd_state.velocities()[0];
        assert!(vel.length() < 1e-6, "Isolated vertex should have zero velocity");
    }
    
    #[test]
    fn test_mesh_tear_state_adjacency() {
        let (mesh, _) = create_tear_test_mesh();
        let tear_state = MeshTearState::with_adjacency(&mesh.faces, &mesh.constraints.edges);
        
        // Edge 0 connects vertices 1 and 2, should be in faces 0 and 1
        let faces = tear_state.faces_for_edge(0);
        assert_eq!(faces.len(), 2, "Edge 0 should be referenced by 2 faces");
        
        // Vertex 1 should have 3 incident edges (0, 2, 3)
        let edges = tear_state.edges_for_vertex(1);
        assert_eq!(edges.len(), 3, "Vertex 1 should have 3 incident edges");
    }
}
