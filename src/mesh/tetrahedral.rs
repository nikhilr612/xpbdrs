//! Tetrahedral mesh implementation.

use raylib::prelude::*;
use std::io::Write;
use tracing::{debug, error, info};

use super::common::{Mesh, Result, TetrahedronId, Triangle, Vertex, dedup_with_warning};
use super::tgimport::TetgenParser;
use crate::constraint::Constraint;
use crate::mesh::{Edge, Tetrahedron};
use crate::xpbd::ConstraintSet;

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
        let _ = processor
            .process(
                self.edges.iter().zip(reference.lengths.iter().copied()),
                params.l_threshold_length,
                params.length_compliance / (params.time_substep * params.time_substep),
                params.shuffle_buffer_size,
            )
            .process(
                self.tetrahedra
                    .iter()
                    .zip(reference.volumes.iter().copied()),
                params.l_threshold_volume,
                params.volume_compliance / (params.time_substep * params.time_substep),
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

    /// Draw wireframe of the mesh.
    pub fn draw_wireframe(&self, d3: &mut RaylibMode3D<RaylibDrawHandle>, color: Color) {
        // Draw explicit edges if available
        for edge in &self.constraints.edges {
            if let (Some(v1), Some(v2)) = (
                self.vertices.get((edge.0.0 - 1) as usize),
                self.vertices.get((edge.1.0 - 1) as usize),
            ) {
                let start = v1.position;
                let end = v2.position;
                d3.draw_line_3D(start, end, color);
            }
        }
    }

    /// Draw filled faces.
    pub fn draw_faces(
        &self,
        d3: &mut RaylibMode3D<RaylibDrawHandle>,
        color: Color,
    ) {
        for face in &self.faces {
            let verts = [
                self.vertices[(face.verts[0].0 - 1) as usize],
                self.vertices[(face.verts[1].0 - 1) as usize],
                self.vertices[(face.verts[2].0 - 1) as usize],
            ];

            d3.draw_triangle3D(
                verts[0].position,
                verts[1].position,
                verts[2].position,
                color,
            );
        }
    }
}

impl Mesh for Tetrahedral {
    fn translate(&mut self, by: Vector3) {
        for vertex in &mut self.vertices {
            vertex.position += by;
        }
    }

    fn bounding_box(&self) -> (Vector3, Vector3) {
        if self.vertices.is_empty() {
            return (Vector3::zero(), Vector3::zero());
        }

        let mut min = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for vertex in &self.vertices {
            min.x = min.x.min(vertex.position.x);
            min.y = min.y.min(vertex.position.y);
            min.z = min.z.min(vertex.position.z);
            max.x = max.x.max(vertex.position.x);
            max.y = max.y.max(vertex.position.y);
            max.z = max.z.max(vertex.position.z);
        }

        (min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        mesh.translate(translation);

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
        let (min, max) = mesh.bounding_box();

        // Based on our test data: vertices at (0,0,0) and (1,1,1)
        assert_eq!(min, Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(max, Vector3::new(1.0, 1.0, 1.0));

        // Cleanup
        for ext in &["node", "edge", "face", "ele"] {
            let _ = fs::remove_file(format!("{prefix}.{ext}"));
        }
    }
}
