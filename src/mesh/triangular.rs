//! Triangulated Surface mesh implementation

use std::collections::HashMap;
use std::io::Write;

use raylib::prelude::*;
use tracing::{debug, info, warn};

use super::common::Result;
use crate::{
    constraint::Constraint,
    mesh::{Edge, EdgeId, Triangle, Vertex, VertexId},
    xpbd::{ConstraintSet, XpbdState},
};

/// Struct to contain edge constraint data for a triangulated mesh.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TriConstraints {
    /// Length constraints for every edge in the mesh.
    edges: Vec<Edge>,
    /// Length constraints between opposite points of adjacent triangles.
    /// This is also an edge, since it a distance constraint between two vertices.
    /// This is a "weak" variant of the more common angle-based bending constraint.
    weak_bending: Vec<Edge>,
}

/// Values computed for the constraints of a triangulated mesh.
pub struct TriConstraintValues {
    /// Lengths for edge constraints.
    pub edge_lengths: Vec<f32>,
    /// Lengths for weak bending constraints.
    pub weak_bending_lengths: Vec<f32>,
}

impl ConstraintSet<Vec<Vertex>, TriConstraintValues> for TriConstraints {
    fn size(&self) -> usize {
        self.edges.len() + self.weak_bending.len()
    }

    fn evaluate(&self, on: &Vec<Vertex>) -> TriConstraintValues {
        let edge_lengths = self.edges.iter().map(|e| e.value(on)).collect();
        let weak_bending_lengths = self.weak_bending.iter().map(|e| e.value(on)).collect();
        TriConstraintValues {
            edge_lengths,
            weak_bending_lengths,
        }
    }

    fn solve(
        &self,
        processor: crate::xpbd::ConstraintProcessor<Vec<Vertex>>,
        params: &crate::xpbd::XpbdParams,
        reference: &TriConstraintValues,
    ) {
        let _ = processor
            .process(
                self.edges
                    .iter()
                    .zip(reference.edge_lengths.iter().copied()),
                params.l_threshold_length,
                params.length_compliance / (params.time_substep * params.time_substep),
                params.shuffle_buffer_size,
            )
            .process(
                self.weak_bending
                    .iter()
                    .zip(reference.weak_bending_lengths.iter().copied()),
                1.1 * params.l_threshold_length, // stronger
                0.9 * params.length_compliance / (params.time_substep * params.time_substep), // stiffer
                params.shuffle_buffer_size,
            );
    }
}

/// A surface described as a triangulated mesh.
/// In contrast with [`mesh::Tetrahedral`], this mesh type does not impose any volume constraints.
/// The triangulated surface instead enforces weak bending constraints between adjacent triangles.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TriangulatedSurface {
    /// Vertices of the triangulated mesh.
    pub vertices: Vec<Vertex>,
    /// Triangular faces of the mesh.
    pub faces: Vec<Triangle>,
    /// Constraints associated with the triangulated mesh.
    pub constraints: TriConstraints,
}

struct EdgeInfo {
    id: EdgeId,
    opposite_vertices: Vec<VertexId>,
}

impl TriangulatedSurface {
    /// Create a new triangulated surface mesh.
    ///
    /// # Panics
    ///
    /// Panics if the number of unique edges exceeds `u32::MAX`.
    #[tracing::instrument(skip(vertices, faces), fields(vertex_count = vertices.len(), face_count = faces.len()))]
    pub fn new(vertices: Vec<Vertex>, faces: &[[VertexId; 3]]) -> Self {
        let dedup_triangles = faces
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut f = *f;
                f.sort_by_key(|v| v.0);
                (f, i)
            })
            .collect::<HashMap<_, _>>();

        // Preserve winding order.
        let triangles = dedup_triangles
            .values()
            .map(|&i| faces[i])
            .collect::<Vec<_>>();

        if faces.len() != triangles.len() {
            warn!(
                "Duplicate faces found in triangulated mesh. Original count: {}, Deduplicated count: {}",
                faces.len(),
                triangles.len()
            );
        }

        let mut edge_map: HashMap<Edge, EdgeInfo> = HashMap::new();

        let mut add_edge = |edge: Edge, opposite_vertex: VertexId| -> EdgeId {
            let size = edge_map
                .len()
                .try_into()
                .expect("Edge count should not exceed u32::MAX");

            let entry = edge_map.entry(edge).or_insert(EdgeInfo {
                id: EdgeId(size),
                opposite_vertices: Vec::new(),
            });
            entry.opposite_vertices.push(opposite_vertex);
            entry.id
        };

        let mut faces = Vec::with_capacity(triangles.len());
        for tri in &triangles {
            let (a, b, c) = {
                let mut c = *tri;
                c.sort_by_key(|v| v.0);
                (c[0], c[1], c[2])
            };

            // Edge(a, b) - opposite vertex is c
            let e1 = add_edge(Edge(a, b), c);
            // Edge(b, c) - opposite vertex is a
            let e2 = add_edge(Edge(b, c), a);
            // Edge(a, c) - opposite vertex is b
            let e3 = add_edge(Edge(a, c), b);
            faces.push(Triangle {
                verts: *tri,
                edges: [Some(e1), Some(e2), Some(e3)],
            });
        }

        let mut edges = vec![Edge(VertexId(0), VertexId(0)); edge_map.len()];
        let mut weak_bending = Vec::new();
        for (edge, info) in edge_map {
            edges[info.id.0 as usize] = edge.clone();
            if info.opposite_vertices.len() >= 2 {
                // Create weak bending constraint between the two opposite vertices
                let vb1 = info.opposite_vertices[0];
                let vb2 = info.opposite_vertices[1];
                weak_bending.push(Edge(vb1, vb2));
                if info.opposite_vertices.len() > 2 {
                    warn!(
                        "More than two triangles share the edge {:?}. Found {} opposite vertices.",
                        edge,
                        info.opposite_vertices.len()
                    );
                }
            }
        }

        let result = Self {
            vertices,
            faces,
            constraints: TriConstraints {
                edges,
                weak_bending,
            },
        };

        info!(
            vertices = result.vertices.len(),
            edges = result.constraints.edges.len(),
            faces = result.faces.len(),
            weak_bending = result.constraints.weak_bending.len(),
            "Triangulated surface mesh created"
        );

        result
    }

    /// Load triangulated mesh from bincode file.
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
        info!("Serializing triangulated surface to binary format");
        let encoded = bincode::serialize(self)?;

        let mut file = std::fs::File::create(output_path)?;
        file.write_all(&encoded)?;

        info!(
            output_path,
            size_bytes = encoded.len(),
            "Successfully exported triangulated surface mesh"
        );

        // Verify deserialization works
        debug!("Verifying serialized data");
        let _: Self = bincode::deserialize(&encoded)?;
        debug!("Verification successful");

        Ok(())
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

    /// Draw weak bending constraints.
    pub fn draw_weak_bending(&self, d3: &mut RaylibMode3D<RaylibDrawHandle>, color: Color) {
        for edge in &self.constraints.weak_bending {
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
    /// # Panics
    /// Panics if any face does not have corresponding edges.
    pub fn draw_faces(
        &self,
        d3: &mut RaylibMode3D<RaylibDrawHandle>,
        state: &XpbdState,
        color: Color,
    ) {
        for face in &self.faces {
            let verts = [
                self.vertices[(face.verts[0].0 - 1) as usize],
                self.vertices[(face.verts[1].0 - 1) as usize],
                self.vertices[(face.verts[2].0 - 1) as usize],
            ];

            // A triangle is "torn" if any of its corresponding edge constraints are inactive.
            let torn = face.edges.iter().any(|e| {
                state.constraint_inactive(
                    e.expect("All triangles should have corresponding edges").0 as usize,
                )
            }); // edges are solved first, so base index is 0.

            if !torn {
                d3.draw_triangle3D(
                    verts[0].position,
                    verts[1].position,
                    verts[2].position,
                    color,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangulated_surface_creation() {
        let vertices = vec![
            Vertex {
                position: Vector3::new(0.0, 0.0, 0.0),
                inv_mass: 1.0,
            },
            Vertex {
                position: Vector3::new(1.0, 0.0, 0.0),
                inv_mass: 1.0,
            },
            Vertex {
                position: Vector3::new(0.5, 1.0, 0.0),
                inv_mass: 1.0,
            },
            Vertex {
                position: Vector3::new(0.5, 0.5, 1.0),
                inv_mass: 1.0,
            },
        ];

        let faces = &[
            [VertexId(1), VertexId(2), VertexId(3)], // Triangle 1
            [VertexId(1), VertexId(2), VertexId(4)], // Triangle 2 (shares edge with Triangle 1)
        ];

        let mesh = TriangulatedSurface::new(vertices, faces);

        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.faces.len(), 2);
        assert_eq!(mesh.constraints.edges.len(), 5); // Each triangle has 3 edges, but one is shared
        assert_eq!(mesh.constraints.weak_bending.len(), 1); // One shared edge creates one bending constraint
    }
}
