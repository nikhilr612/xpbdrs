//! Triangulated Surface mesh implementation

use crate::{
    constraint::Constraint,
    mesh::{Edge, Triangle, Vertex},
    xpbd::ConstraintSet,
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
                1.1 * params.l_threshold_length,
                params.length_compliance / (params.time_substep * params.time_substep),
                params.shuffle_buffer_size,
            );
    }
}

/// A surface described as a triangulated mesh.
/// In contrast with [`mesh::Tetrahedral`], this mesh type does not impose any volume constraints.
/// The triangulated surface instead enforces weak bending constraints between adjacent triangles.
pub struct TriangulatedSurface {
    /// Vertices of the triangulated mesh.
    vertices: Vec<Vertex>,
    /// Triangular faces of the mesh.
    faces: Vec<Triangle>,
    /// Constraints associated with the triangulated mesh.
    constraints: TriConstraints,
}
