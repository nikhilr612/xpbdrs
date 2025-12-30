//! Cloth simulation demo using triangulated surface mesh.

use clap::Parser;
use raylib::prelude::*;
use tracing::info;

use xpbdrs::{
    mesh::{Spatial, TriangulatedSurface, Vertex, VertexId},
    xpbd::{self, ConstraintSet, XpbdState},
};

#[derive(Parser)]
#[command(name = "cloth")]
#[command(about = "XPBD cloth simulation with triangulated surface")]
#[command(version)]
struct Cli {
    /// Grid resolution (number of divisions along each axis)
    #[arg(short, long, default_value = "20")]
    resolution: usize,

    /// Size of the cloth in world units
    #[arg(short, long, default_value = "5.0")]
    size: f32,
}

const TARGET_FPS: u16 = 60;
const TIME_STEP: f32 = 1.0 / TARGET_FPS as f32;
const N_SUBSTEPS: usize = 15;
const EDGE_COMPLIANCE: f32 = 0.001;
const TEARING_THRESHOLD: f32 = 2500.0;
const BOUNDARY_INV_MASS: f32 = 0.0001;
const REGULAR_INV_MASS: f32 = 1.0;

// Demo parameters
const SPAWN_HEIGHT: f32 = 2.0;
const PUNCTURE_START_TIME: f32 = 3.0;
const PUNCTURE_DURATION: f32 = 1.0;
const PUNCTURE_FORCE: f32 = 100.0;
const PUNCTURE_RADIUS: f32 = 0.5;

fn generate_cloth_mesh(resolution: usize, size: f32) -> TriangulatedSurface {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    let step = size / resolution as f32;
    let half_size = size * 0.5;

    // Generate vertices
    for i in 0..=resolution {
        for j in 0..=resolution {
            let x = -half_size + i as f32 * step;
            let z = -half_size + j as f32 * step;
            let y = SPAWN_HEIGHT;

            let is_boundary = i == 0 || i == resolution || j == 0 || j == resolution;
            let inv_mass = if is_boundary {
                BOUNDARY_INV_MASS
            } else {
                REGULAR_INV_MASS
            };

            vertices.push(Vertex {
                position: Vector3::new(x, y, z),
                inv_mass,
            });
        }
    }

    // Generate triangular faces
    for i in 0..resolution {
        for j in 0..resolution {
            let bottom_left = i * (resolution + 1) + j;
            let bottom_right = bottom_left + 1;
            let top_left = (i + 1) * (resolution + 1) + j;
            let top_right = top_left + 1;

            // Two triangles per quad (counter-clockwise winding)
            faces.push([
                VertexId((bottom_left + 1) as u32),
                VertexId((bottom_right + 1) as u32),
                VertexId((top_left + 1) as u32),
            ]);

            faces.push([
                VertexId((bottom_right + 1) as u32),
                VertexId((top_right + 1) as u32),
                VertexId((top_left + 1) as u32),
            ]);
        }
    }

    TriangulatedSurface::new(vertices, &faces)
}

fn setup_camera(mesh: &TriangulatedSurface) -> (Vector3, Vector3) {
    let (min, max) = mesh.vertices.bounding_box();
    let center = (min + max) * 0.5;
    let distance = (max - min).length().max(1.0) * 1.5;
    let camera_pos = center + Vector3::new(distance * 0.7, distance * 0.5, distance * 0.7);
    (camera_pos, center)
}

fn handle_input(
    rl: &RaylibHandle,
    show_wireframe: &mut bool,
    show_faces: &mut bool,
    should_reset: &mut bool,
    camera: &mut Camera3D,
) {
    if rl.is_key_pressed(KeyboardKey::KEY_X) {
        *show_wireframe = !*show_wireframe;
    }
    if rl.is_key_pressed(KeyboardKey::KEY_F) {
        *show_faces = !*show_faces;
    }
    if rl.is_key_pressed(KeyboardKey::KEY_R) {
        *should_reset = true;
    }

    // Camera movement
    const CAMERA_SPEED: f32 = 0.2;
    if rl.is_key_down(KeyboardKey::KEY_SPACE) {
        camera.position.y += CAMERA_SPEED;
        camera.target.y += CAMERA_SPEED;
    } else if rl.is_key_down(KeyboardKey::KEY_LEFT_SHIFT) {
        camera.position.y -= CAMERA_SPEED;
        camera.target.y -= CAMERA_SPEED;
    }
}

fn run_simulation(resolution: usize, size: f32) {
    let mut mesh = generate_cloth_mesh(resolution, size);
    let original_mesh = mesh.clone();
    let original_positions: Vec<Vector3> = mesh.vertices.iter().map(|v| v.position).collect();

    let mut show_wireframe = true;
    let mut show_faces = true;
    let mut should_reset = false;
    let mut simulation_time = 0.0;

    let (mut rl, thread) = raylib::init()
        .size(1200, 800)
        .title("XPBD Cloth Simulation")
        .build();

    let (camera_pos, target) = setup_camera(&mesh);
    let mut camera = Camera3D::perspective(camera_pos, target, Vector3::new(0.0, 1.0, 0.0), 60.0);
    rl.set_target_fps(TARGET_FPS.into());

    let initial_values = mesh.constraints.evaluate(&mesh.vertices);
    let xpbd_params = xpbd::XpbdParams {
        n_substeps: N_SUBSTEPS,
        time_substep: TIME_STEP / (N_SUBSTEPS as f32),
        length_compliance: EDGE_COMPLIANCE,
        volume_compliance: 0.0,
        l_threshold_length: TEARING_THRESHOLD,
        l_threshold_volume: f32::INFINITY,
        p_volume: 1.0,
    };

    let mut state = XpbdState::new(mesh.vertices.len(), mesh.constraints.size());

    info!(
        vertices = mesh.vertices.len(),
        constraints = mesh.constraints.size(),
        resolution = resolution,
        size = size,
        "Starting cloth simulation"
    );

    while !rl.window_should_close() {
        handle_input(
            &rl,
            &mut show_wireframe,
            &mut show_faces,
            &mut should_reset,
            &mut camera,
        );
        rl.update_camera(&mut camera, CameraMode::CAMERA_THIRD_PERSON);

        if should_reset {
            mesh = original_mesh.clone();
            state = XpbdState::new(mesh.vertices.len(), mesh.constraints.size());
            simulation_time = 0.0;
            should_reset = false;
        }

        simulation_time += TIME_STEP;

        // Physics simulation
        let cloth_center = mesh.vertices.centroid();
        let acceleration_field = |vertex: &Vertex| {
            if vertex.inv_mass <= BOUNDARY_INV_MASS * 1.1 {
                Vector3::zero() // Fixed boundary vertices
            } else {
                let mut acceleration = Vector3::new(0.0, -9.81, 0.0); // Gravity

                // Puncturing force
                if (PUNCTURE_START_TIME..PUNCTURE_START_TIME + PUNCTURE_DURATION)
                    .contains(&simulation_time)
                {
                    let distance = (vertex.position - cloth_center).length();
                    if distance < PUNCTURE_RADIUS {
                        let force_factor = (PUNCTURE_RADIUS - distance) / PUNCTURE_RADIUS;
                        acceleration.y -= PUNCTURE_FORCE * force_factor;
                    }
                }

                acceleration
            }
        };

        for _ in 0..xpbd_params.n_substeps {
            xpbd::substep(
                &xpbd_params,
                &mut state,
                &mut mesh.vertices,
                &mesh.constraints,
                &initial_values,
                &mut |_v| {},
                &acceleration_field,
            );
        }

        // Apply boundary constraints
        for (i, vertex) in mesh.vertices.iter_mut().enumerate() {
            if vertex.inv_mass <= BOUNDARY_INV_MASS * 1.1 {
                vertex.position = original_positions[i];
            }
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::new(205, 206, 245, 255));

        {
            let mut d3 = d.begin_mode3D(camera);

            // Ground
            d3.draw_plane(
                Vector3::new(0.0, -0.1, 0.0),
                Vector2::new(50.0, 50.0),
                Color::new(50, 50, 50, 255),
            );
            d3.draw_plane(
                Vector3::new(0.0, 0.0, 0.0),
                Vector2::new(10.0, 10.0),
                Color::new(70, 70, 70, 255),
            );
            d3.draw_grid(20, 1.0);

            // Cloth
            if show_faces {
                mesh.draw_faces(&mut d3, &state, Color::new(255, 255, 255, 200));
            }
            if show_wireframe {
                mesh.draw_wireframe(&mut d3, Color::new(30, 144, 255, 255));
            }
        }

        d.draw_fps(10, 10);
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    run_simulation(cli.resolution, cli.size);
}
