//! Implement xpbd on a tetrahedral mesh.

use std::ops::IndexMut;

use raylib::math::Vector3;

use crate::{
    constraint::{Constraint, ValueGrad, apply_constraint},
    mesh::{Tetrahedral, Vertex, VertexId},
};

/// State for Extended Position Based Dynamics simulation.
pub struct XpbdState {
    /// Velocities of each particle.
    velocities: Vec<Vector3>,
}

/// Immutable parameters for the XPBD simulation.
#[derive(Clone, Debug)]
pub struct XpbdParams {
    /// Stiffness for edge length constraints.
    stiffness_volume: f32,
    /// Stiffness for tetrahedral volume constraints.
    stiffness_length: f32,
    /// Number of substeps per simulation step.
    n_substeps: usize,
    /// Time step for each simulation substep.
    time_substep: f32,
}

impl XpbdParams {
    /// Create new XPBD parameters.
    pub fn new(
        n_substeps: usize,
        time_step: f32,
        stiffness_length: f32,
        stiffness_volume: f32,
    ) -> Self {
        Self {
            stiffness_length,
            stiffness_volume,
            n_substeps,
            time_substep: time_step / n_substeps as f32,
        }
    }
}

impl XpbdState {
    /// Initialize the XPBD state with given number of vertices, substeps, and time step.
    pub fn new(n_vertices: usize) -> Self {
        Self {
            velocities: vec![Vector3::zero(); n_vertices],
        }
    }
}

pub struct TetConstraintValues {
    lengths: Vec<f32>,
    volumes: Vec<f32>,
}

pub fn evaluate_tet_constraints(mesh: &Tetrahedral) -> TetConstraintValues {
    let lengths = mesh.edges.iter().map(|e| e.value(&mesh.vertices)).collect();
    let volumes = mesh
        .tetrahedra
        .iter()
        .map(|t| t.value(&mesh.vertices))
        .collect();
    TetConstraintValues { lengths, volumes }
}

// TODO: Implement more generic Xpbd function.
pub fn step_basic(
    params: &XpbdParams,
    state: XpbdState,
    mesh: &mut Tetrahedral,
    initial_value: &TetConstraintValues,
) -> XpbdState {
    let XpbdState { mut velocities } = state;
    let XpbdParams {
        stiffness_volume,
        stiffness_length,
        n_substeps,
        time_substep,
    } = params.clone();
    for _ in 0..n_substeps {
        // copy old positions each time.
        let old_positions = mesh.vertices.clone();
        let gravity = Vector3::new(0.0, -0.1, 0.0);

        for (i, vertex) in mesh.vertices.iter_mut().enumerate() {
            velocities[i] += gravity * time_substep; // unit mass for now
            vertex.position += velocities[i] * time_substep;
            if vertex.position.y < 0.0 {
                vertex.position.y = 0.0;
            }
        }

        for (edge, &ref_length) in mesh.edges.iter().zip(initial_value.lengths.iter()) {
            let result = edge.value_and_grad(&mesh.vertices);
            let alpha = stiffness_length / (time_substep * time_substep);
            apply_constraint(result, ref_length, alpha, &mut mesh.vertices);
        }

        for (i, tetrahedron) in mesh.tetrahedra.iter().enumerate() {
            let ref_volume = *initial_value
                .volumes
                .get(i)
                .expect("Tetrahedron should have an initial volume.");
            let result = tetrahedron.value_and_grad(&mesh.vertices);
            let alpha = stiffness_volume / (time_substep * time_substep);
            apply_constraint(result, ref_volume, alpha, &mut mesh.vertices);
        }

        // Update velocities based on position changes
        for (i, vertex) in mesh.vertices.iter().enumerate() {
            velocities[i] = (vertex.position - old_positions[i].position) / time_substep;
        }
    }
    XpbdState { velocities }
}
