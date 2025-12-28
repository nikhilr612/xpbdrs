//! Implement xpbd on a tetrahedral mesh.

use std::ops::IndexMut;

use bitvec::vec::BitVec;
use rand::seq::SliceRandom;
use raylib::math::Vector3;

use crate::{
    constraint::{Constraint, apply_constraint},
    mesh::{Tetrahedral, Vertex, VertexId, tetrahedral::TetConstraintValues},
};

/// State for Extended Position Based Dynamics simulation.
pub struct XpbdState {
    /// Velocities of each particle.
    velocities: Vec<Vector3>,
    /// Boolean vector indicating inactive constraints by index.
    inactive_constraints: BitVec,
    /// Vector to store old positions during substeps.
    position_buffer: Vec<Vector3>,
}

impl XpbdState {
    #[must_use]
    /// Check if a constraint at given index is inactive.
    /// Note that constraints are indexed in the order they are processed during constraint solving (see [`ConstraintProcessor`]).
    /// So, index-0 corresponds to the first constraint processed, index-1 to the second, and so on.
    /// If the index is out of bounds, the constraint is considered active (i.e., returns false).
    pub fn constraint_inactive(&self, index: usize) -> bool {
        self.inactive_constraints
            .as_bitslice()
            .get(index)
            .is_some_and(|b| *b)
    }
}

/// Immutable parameters for the XPBD simulation.
#[derive(Clone, Debug)]
pub struct XpbdParams {
    /// A parameter that is inversely proportional to stiffness for volume constraints.
    /// In particular, a value of 0.0 corresponds to infinite stiffness.
    pub volume_compliance: f32,
    /// A parameter that is inversely proportional to stiffness for edge length constraints.
    /// In particular, a value of 0.0 corresponds to infinite stiffness.
    pub length_compliance: f32,
    /// Damping factor applied to velocities (0.0 = no damping, 1.0 = full damping).
    pub damping: f32,
    /// Number of substeps per simulation step.
    pub n_substeps: usize,
    /// Time step for each simulation substep.
    pub time_substep: f32,
    /// Length constraint force-threshold for deactivation.
    pub l_threshold_length: f32,
    /// Volume constraint force-threshold for deactivation.
    pub l_threshold_volume: f32,
    /// Size of the shuffle buffer for constraint processing.
    /// Constraints are collected into chunks of this size and shuffled before processing.
    /// Use `usize::MAX` to shuffle all constraints together (full shuffle).
    /// Use `1` to disable shuffling entirely.
    pub shuffle_buffer_size: usize,
    /// Constant acceleration field applied to all vertices (e.g., gravity).
    pub constant_field: Vector3,
}

impl Default for XpbdParams {
    fn default() -> Self {
        Self {
            length_compliance: 0.0,
            volume_compliance: 0.0,
            damping: 0.0,
            n_substeps: 10,
            time_substep: 0.016 / 10.0,
            l_threshold_length: f32::INFINITY,
            l_threshold_volume: f32::INFINITY,
            shuffle_buffer_size: usize::MAX, // Full shuffle by default
            constant_field: Vector3::new(0.0, -9.81, 0.0), // Default gravity
        }
    }
}

impl XpbdState {
    /// Initialize the XPBD state with given number of vertices, substeps, and time step.
    #[must_use]
    pub fn new(n_vertices: usize, n_constraints: usize) -> Self {
        Self {
            velocities: vec![Vector3::zero(); n_vertices],
            position_buffer: vec![Vector3::zero(); n_vertices],
            inactive_constraints: BitVec::repeat(false, n_constraints),
        }
    }
}

/// Helper struct to solve constraints
pub struct ConstraintProcessor<'solver, V: IndexMut<VertexId, Output = Vertex>> {
    inactive_constraints: &'solver mut BitVec,
    vertices: &'solver mut V,
    constraint_index: usize,
}

impl<V: IndexMut<VertexId, Output = Vertex>> ConstraintProcessor<'_, V> {
    /// Process constraints from an iterator, applying them to the vertices and deactivating those that exceed the threshold.
    /// Internally, the index of each constraint in the iterator is used to track constraint active status.
    /// Thus, it is imperative that `process` is called each time with constraints in the same order.
    ///
    /// Constraints are shuffled on-the-fly in chunks of `buffer_size` for better convergence
    /// (avoids systematic bias from fixed iteration order).
    #[must_use]
    pub fn process<'a, I, C, const N: usize>(
        mut self,
        iter: I,
        l_threshold: f32,
        alpha: f32,
        buffer_size: usize,
    ) -> Self
    where
        I: Iterator<Item = (&'a C, f32)>,
        C: Constraint<N> + 'a,
    {
        let base_index = self.constraint_index;
        let mut rng = rand::thread_rng();
        let mut total_count = 0;

        // Process constraints in shuffled chunks
        let mut peekable = iter.enumerate().peekable();
        while peekable.peek().is_some() {
            let mut buffer: Vec<(usize, (&C, f32))> = peekable.by_ref().take(buffer_size).collect();
            total_count += buffer.len();
            buffer.shuffle(&mut rng);

            for (i, (constraint, ref_value)) in buffer {
                let current_index = base_index + i;
                if !self.inactive_constraints[current_index] {
                    let result = constraint.value_and_grad(self.vertices);
                    if apply_constraint(result, ref_value, alpha, self.vertices) > l_threshold {
                        self.inactive_constraints.set(current_index, true);
                    }
                }
            }
        }

        self.constraint_index = base_index + total_count;
        self
    }
}

/// Basic XPBD step function for tetrahedral meshes.
/// Additionally accepts a vertex correction function to handle collisions and other vertex corrections that need to be applied after the kinematic update.
pub fn step_basic<F>(
    params: &XpbdParams,
    mut state: XpbdState,
    mesh: &mut Tetrahedral,
    initial_value: &TetConstraintValues,
    mut vertex_correction: F,
) -> XpbdState
where
    F: FnMut(&mut Vertex),
{
    let constant_field = params.constant_field;
    let acceleration_field = move |_: &Vertex| constant_field;
    for _ in 0..params.n_substeps {
        substep(
            params,
            &mut state,
            &mut mesh.vertices,
            &mesh.constraints,
            initial_value,
            &mut vertex_correction,
            &acceleration_field,
        );
    }
    state
}

/// Trait for constraint set over vertices collected in `V`, evaluating to constraint errors of type `I`.
pub trait ConstraintSet<V: IndexMut<VertexId, Output = Vertex>, I> {
    /// Evaluate the constraint set on given vertices.
    fn evaluate(&self, on: &V) -> I;
    /// Solve the constraint set using the given processor.
    fn solve(&self, processor: ConstraintProcessor<V>, params: &XpbdParams, reference: &I);
}

/// Perform a single substep of XPBD simulation.
/// This includes kinematic updates, constraint solving, and velocity updates.
/// The `acceleration_field` closure allows for flexible force application (e.g., gravity, wind, etc.) on each vertex.
/// The `post_kinematic_correction` closure allows for custom vertex corrections after the kinematic update (e.g., collision handling).
pub fn substep<V, I, F, C, A>(
    params: &XpbdParams,
    state: &mut XpbdState,
    vertices: &mut V,
    constraint_set: &C,
    initial_value: &I,
    post_kinematic_correction: &mut F,
    acceleration_field: &A,
) where
    C: ConstraintSet<V, I>,
    V: IndexMut<VertexId, Output = Vertex>,
    for<'a> &'a mut V: IntoIterator<Item = &'a mut Vertex>,
    F: FnMut(&mut Vertex),
    A: Fn(&Vertex) -> Vector3,
{
    let old_positions = &mut state.position_buffer; // use buffer for old positions.
    let damping_factor = 1.0 - params.damping;

    for (i, vertex) in vertices.into_iter().enumerate() {
        // save old position
        old_positions[i] = vertex.position;

        // Apply damping to velocity
        state.velocities[i] *= damping_factor;
        state.velocities[i] += acceleration_field(vertex) * params.time_substep;
        vertex.position += state.velocities[i] * params.time_substep;

        post_kinematic_correction(vertex);
    }

    let processor = ConstraintProcessor {
        inactive_constraints: &mut state.inactive_constraints,
        vertices,
        constraint_index: 0,
    };
    constraint_set.solve(processor, params, initial_value);

    // Update velocities based on position changes
    for (i, vertex) in vertices.into_iter().enumerate() {
        let new_velocity = (vertex.position - old_positions[i]) / params.time_substep;
        // Update velocity in state
        state.velocities[i] = new_velocity;
    }
}
