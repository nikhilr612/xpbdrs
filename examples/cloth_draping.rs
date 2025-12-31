//! Cloth draping simulation demo using triangulated surface mesh.

use clap::Parser;
use raylib::prelude::*;
use tracing::info;

use xpbdrs::{
    mesh::{TriangulatedSurface, Vertex, VertexId},
    xpbd::{self, ConstraintSet, XpbdState},
};

#[derive(Parser)]
#[command(name = "cloth_draping")]
#[command(about = "XPBD cloth draping simulation with triangulated surface")]
#[command(version)]
struct Cli {
    /// Cloth grid resolution (number of divisions along each axis)
    #[arg(long, default_value = "25")]
    resolution: usize,

    /// Size of the cloth in world units
    #[arg(long, default_value = "4.0")]
    size: f32,

    /// Height at which to spawn the cloth above the sphere
    #[arg(long, default_value = "2.5")]
    height: f32,

    /// Radius of the sphere to drape over
    #[arg(long, default_value = "1.0")]
    sphere_radius: f32,
}

const TARGET_FPS: u16 = 60;
const TIME_STEP: f32 = 1.0 / TARGET_FPS as f32;
const N_SUBSTEPS: usize = 10;
const EDGE_COMPLIANCE: f32 = 0.0001;
const TEARING_THRESHOLD: f32 = 3000.0;

// Cloth physics parameters
const CLOTH_MASS: f32 = 1.0; // Same mass for all cloth vertices
const CLOTH_INV_MASS: f32 = 1.0 / CLOTH_MASS;

// Sphere collision parameters
const SPHERE_CENTER: glam::Vec3 = glam::Vec3::new(0.0, 0.0, 0.0);
const COLLISION_MARGIN: f32 = 0.05;

fn generate_cloth_mesh(resolution: usize, size: f32, spawn_height: f32) -> TriangulatedSurface {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    let step = size / resolution as f32;
    let half_size = size * 0.5;

    // Generate vertices - all vertices have the same mass
    for i in 0..=resolution {
        for j in 0..=resolution {
            let x = -half_size + i as f32 * step;
            let z = -half_size + j as f32 * step;
            let y = spawn_height;

            vertices.push(Vertex {
                position: glam::Vec3::new(x, y, z),
                inv_mass: CLOTH_INV_MASS,
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

fn setup_camera(cloth_size: f32, sphere_radius: f32) -> (Vector3, Vector3) {
    let max_extent = cloth_size.max(sphere_radius * 2.0);
    let distance = max_extent * 2.0;
    let camera_pos = Vector3::new(distance * 0.8, distance * 0.6, distance * 0.8);
    let center = Vector3::new(0.0, sphere_radius * 0.5, 0.0);
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

fn run_simulation(resolution: usize, size: f32, spawn_height: f32, sphere_radius: f32) {
    let mut mesh = generate_cloth_mesh(resolution, size, spawn_height);
    let original_mesh = mesh.clone();

    let mut show_wireframe = true;
    let mut show_faces = true;

    let mut should_reset = false;

    let (mut rl, thread) = raylib::init()
        .size(800, 800)
        .title("XPBD Cloth Draping Simulation")
        .build();

    let (camera_pos, target) = setup_camera(size, sphere_radius);
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
        cloth_size = size,
        spawn_height = spawn_height,
        sphere_radius = sphere_radius,
        "Starting cloth draping simulation"
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
            should_reset = false;
        }

        // Physics simulation
        let acceleration_field = |_vertex: &Vertex| {
            glam::Vec3::new(0.0, -9.81, 0.0) // Gravity
        };

        let mut collision = |vertex: &mut Vertex| {
            let to_vertex = vertex.position - SPHERE_CENTER;
            let distance = to_vertex.length();

            if distance < sphere_radius + COLLISION_MARGIN {
                // Push vertex out of the sphere
                let normal = if distance > 0.001 {
                    to_vertex / distance
                } else {
                    glam::Vec3::new(0.0, 1.0, 0.0) // Default upward normal
                };
                vertex.position = SPHERE_CENTER + normal * (sphere_radius + COLLISION_MARGIN);
            }

            vertex.position.y = vertex.position.y.max(-1.0); // Ground plane at y = -1.0
        };

        for _ in 0..xpbd_params.n_substeps {
            xpbd::substep(
                &xpbd_params,
                &mut state,
                &mut mesh.vertices,
                &mesh.constraints,
                &initial_values,
                &mut collision,
                &acceleration_field,
            );
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::new(205, 206, 245, 255));

        {
            let mut d3 = d.begin_mode3D(camera);

            // Ground plane
            d3.draw_plane(
                Vector3::new(0.0, -2.0, 0.0),
                Vector2::new(50.0, 50.0),
                Color::new(50, 50, 50, 255),
            );
            d3.draw_grid(20, 1.0);

            // Sphere to drape over
            const RAY_SPHERE_CENTER: Vector3 =
                Vector3::new(SPHERE_CENTER.x, SPHERE_CENTER.y, SPHERE_CENTER.z);
            d3.draw_sphere(
                RAY_SPHERE_CENTER,
                sphere_radius,
                Color::new(200, 100, 100, 200),
            );
            d3.draw_sphere_wires(
                RAY_SPHERE_CENTER,
                sphere_radius,
                16,
                16,
                Color::new(150, 50, 50, 255),
            );

            // Cloth mesh
            if show_faces {
                mesh.draw_faces(&mut d3, &state, Color::new(255, 255, 255, 180));
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
    run_simulation(cli.resolution, cli.size, cli.height, cli.sphere_radius);
}
