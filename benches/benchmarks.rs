use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use glam::Vec3;
use xpbdrs::{
    constraint::{Constraint, apply_constraint},
    mesh::{Tetrahedral, tetrahedral::TetConstraintValues},
    xpbd::{XpbdParams, XpbdState, step_basic, substep},
};

fn load_test_mesh() -> Tetrahedral {
    Tetrahedral::from_bincode("mesh/utah_teapot.bin")
        .expect("Required test mesh 'mesh/utah_teapot.bin' not found. Cannot run benchmarks without realistic mesh data.")
}

fn create_initial_values(mesh: &Tetrahedral) -> TetConstraintValues {
    TetConstraintValues {
        lengths: mesh
            .constraints
            .edges
            .iter()
            .map(|e| e.value(&mesh.vertices))
            .collect(),
        volumes: mesh
            .constraints
            .tetrahedra
            .iter()
            .map(|t| t.value(&mesh.vertices))
            .collect(),
    }
}

// Core constraint computation benchmarks
fn edge_constraint_computation(c: &mut Criterion) {
    let mesh = load_test_mesh();

    c.bench_function("edge_constraints", |b| {
        b.iter(|| {
            for edge in &mesh.constraints.edges {
                black_box(edge.value_and_grad(&mesh.vertices));
            }
        })
    });
}

fn tetrahedron_constraint_computation(c: &mut Criterion) {
    let mesh = load_test_mesh();

    c.bench_function("tetrahedron_constraints", |b| {
        b.iter(|| {
            for tet in &mesh.constraints.tetrahedra {
                black_box(tet.value_and_grad(&mesh.vertices));
            }
        })
    });
}

fn constraint_application(c: &mut Criterion) {
    let mesh = load_test_mesh();

    c.bench_function("constraint_application", |b| {
        b.iter_batched(
            || mesh.vertices.clone(),
            |mut vertices| {
                if !mesh.constraints.edges.is_empty() {
                    let vag = mesh.constraints.edges[0].value_and_grad(&vertices);
                    black_box(apply_constraint(vag, 0.1, 0.0, 0.01, &mut vertices));
                }
                if !mesh.constraints.tetrahedra.is_empty() {
                    let vag = mesh.constraints.tetrahedra[0].value_and_grad(&vertices);
                    black_box(apply_constraint(vag, 0.01, 0.0, 0.01, &mut vertices));
                }
            },
            BatchSize::SmallInput,
        )
    });
}

fn constraint_solving_iteration(c: &mut Criterion) {
    let mesh = load_test_mesh();

    c.bench_function("constraint_solving_iteration", |b| {
        b.iter_batched(
            || mesh.vertices.clone(),
            |mut vertices| {
                for edge in &mesh.constraints.edges {
                    let vag = edge.value_and_grad(&vertices);
                    black_box(apply_constraint(vag, 0.1, 0.0, 0.01, &mut vertices));
                }
                for tet in &mesh.constraints.tetrahedra {
                    let vag = tet.value_and_grad(&vertices);
                    black_box(apply_constraint(vag, 0.01, 0.0, 0.01, &mut vertices));
                }
            },
            BatchSize::SmallInput,
        )
    });
}

// XPBD simulation benchmarks
fn xpbd_substep(c: &mut Criterion) {
    let mesh = load_test_mesh();
    let initial_values = create_initial_values(&mesh);
    let params = XpbdParams::default();

    c.bench_function("xpbd_substep", |b| {
        b.iter_batched(
            || {
                let mesh = mesh.clone();
                let state = XpbdState::new(
                    mesh.vertices.len(),
                    mesh.constraints.edges.len() + mesh.constraints.tetrahedra.len(),
                );
                (mesh, state)
            },
            |(mut mesh, mut state)| {
                substep(
                    &params,
                    &mut state,
                    &mut mesh.vertices,
                    &mesh.constraints,
                    &initial_values,
                    &mut |_| {},
                    &|_| Vec3::new(0.0, -9.81, 0.0),
                );
            },
            BatchSize::SmallInput,
        )
    });
}

fn xpbd_full_step(c: &mut Criterion) {
    let mesh = load_test_mesh();
    let initial_values = create_initial_values(&mesh);
    let params = XpbdParams {
        n_substeps: 5,
        ..Default::default()
    };

    c.bench_function("xpbd_full_step", |b| {
        b.iter_batched(
            || {
                let mesh = mesh.clone();
                let state = XpbdState::new(
                    mesh.vertices.len(),
                    mesh.constraints.edges.len() + mesh.constraints.tetrahedra.len(),
                );
                (mesh, state)
            },
            |(mut mesh, state)| {
                black_box(step_basic(
                    &params,
                    state,
                    &mut mesh,
                    &initial_values,
                    |_| {},
                ));
            },
            BatchSize::SmallInput,
        )
    });
}

fn kinematic_integration(c: &mut Criterion) {
    let mesh = load_test_mesh();

    c.bench_function("kinematic_integration", |b| {
        b.iter_batched(
            || {
                let vertices = mesh.vertices.clone();
                let velocities = vec![Vec3::new(0.1, 0.0, 0.0); vertices.len()];
                (vertices, velocities)
            },
            |(mut vertices, mut velocities)| {
                let dt = 0.0016;
                let gravity = Vec3::new(0.0, -9.81, 0.0);

                for (i, vertex) in vertices.iter_mut().enumerate() {
                    velocities[i] += gravity * dt;
                    vertex.position += velocities[i] * dt;
                }
                black_box((vertices, velocities));
            },
            BatchSize::SmallInput,
        )
    });
}

// Memory access pattern benchmarks
fn memory_access_patterns(c: &mut Criterion) {
    let mesh = load_test_mesh();

    // Sequential vertex access
    c.bench_function("sequential_vertex_access", |b| {
        b.iter(|| {
            let mut sum = Vec3::ZERO;
            for vertex in &mesh.vertices {
                sum += vertex.position;
            }
            black_box(sum);
        })
    });

    // Constraint-driven vertex access (follows edge connectivity)
    c.bench_function("constraint_vertex_access", |b| {
        b.iter(|| {
            for edge in &mesh.constraints.edges {
                let v1 = &mesh.vertices[edge.0];
                let v2 = &mesh.vertices[edge.1];
                black_box((v1.position, v2.position));
            }
        })
    });

    // Mesh cloning cost (memory allocation)
    c.bench_function("mesh_clone_cost", |b| {
        b.iter_batched(
            || &mesh,
            |mesh| {
                black_box(mesh.vertices.clone());
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    benches,
    edge_constraint_computation,
    tetrahedron_constraint_computation,
    constraint_application,
    constraint_solving_iteration,
    xpbd_substep,
    xpbd_full_step,
    kinematic_integration,
    memory_access_patterns,
);

criterion_main!(benches);
