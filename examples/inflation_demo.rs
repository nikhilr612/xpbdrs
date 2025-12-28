//! Inflation demo scene that spawns a deci_spot mesh at height and inflates it over time.

use clap::Parser;
use raylib::prelude::*;
use tracing::info;

use xpbdrs::{
    mesh::{self, Spatial},
    xpbd::{self, ConstraintSet, XpbdState},
};

#[derive(Parser)]
#[command(name = "inflation_demo")]
#[command(about = "XPBD inflation demo with deci_spot mesh")]
#[command(version)]
struct Cli {
    /// Optional custom mesh file prefix to visualize instead of deci_spot
    mesh: Option<String>,
}

const TARGET_FPS: u16 = 60;
const TIME_STEP: f32 = 1.0 / TARGET_FPS as f32;
const N_SUBSTEPS: usize = 10;
const EDGE_COMPLIANCE: f32 = 0.000;
const VOLUME_COMPLIANCE: f32 = 0.000;
const TEARING_THRESHOLD: f32 = 12500.0; // Force threshold for edge tearing

// Demo parameters
const SPAWN_HEIGHT: f32 = 8.0;
const INFLATION_START_TIME: f32 = 2.0; // Start inflating after 2 seconds
const INFLATION_DURATION: f32 = 5.0; // Inflate over 5 seconds
const INITIAL_DEFLATION: f32 = 0.7; // Start deflated at 70% volume
const MAX_INFLATION: f32 = 2.5; // Maximum inflation factor

fn setup_camera(mesh: Option<&mesh::Tetrahedral>) -> (Vector3, Vector3) {
    mesh.map_or(
        (Vector3::new(7.0, 7.0, 7.0), Vector3::new(0.0, 0.0, 0.0)),
        |mesh| {
            let (min, max) = mesh.vertices.bounding_box();
            let center = (min + max) * 0.5;
            let distance = (max - min).length().max(1.0) * 2.5;
            let camera_pos = center + Vector3::one() * distance * 0.7;
            (camera_pos, center)
        },
    )
}

fn handle_input(
    rl: &RaylibHandle,
    show_wireframe: &mut bool,
    show_faces: &mut bool,
    should_reset: &mut bool,
    camera: &mut Camera3D,
    paused: &mut bool,
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
    if rl.is_key_pressed(KeyboardKey::KEY_P) {
        *paused = !*paused;
    }

    // Camera Y-axis movement
    const CAMERA_SPEED: f32 = 0.2;
    if rl.is_key_down(KeyboardKey::KEY_SPACE) {
        camera.position.y += CAMERA_SPEED;
        camera.target.y += CAMERA_SPEED;
    } else if rl.is_key_down(KeyboardKey::KEY_LEFT_SHIFT)
        || rl.is_key_down(KeyboardKey::KEY_RIGHT_SHIFT)
    {
        camera.position.y -= CAMERA_SPEED;
        camera.target.y -= CAMERA_SPEED;
    }
}

fn draw_mesh(
    d3: &mut RaylibMode3D<RaylibDrawHandle>,
    mesh: &mesh::Tetrahedral,
    state: &XpbdState,
    show_wireframe: bool,
    show_faces: bool,
) {
    if show_faces {
        mesh.draw_faces(d3, state, Color::new(255, 215, 0, 200));
    }
    if show_wireframe {
        mesh.draw_wireframe(d3, Color::new(30, 144, 255, 255));
    }
}

fn draw_ui(d: &mut RaylibDrawHandle) {
    d.draw_fps(10, 10);
}

fn load_mesh(mesh_path: Option<&str>) -> Option<mesh::Tetrahedral> {
    let path = mesh_path.unwrap_or("mesh/deci_spot.bin");
    info!(path, "Loading mesh");

    let load_result = if std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
    {
        mesh::Tetrahedral::from_bincode(path)
    } else {
        mesh::Tetrahedral::from_files(path)
    };

    load_result
        .map(|mut m| {
            m.vertices.translate(Vector3::new(0.0, SPAWN_HEIGHT, 0.0));
            info!(
                vertices = m.vertices.len(),
                edges = m.constraints.edges.len(),
                tetrahedra = m.constraints.tetrahedra.len(),
                "Mesh loaded at height {}",
                SPAWN_HEIGHT
            );
            m
        })
        .ok()
}

fn calculate_inflation_factor(simulation_time: f32) -> f32 {
    if simulation_time < INFLATION_START_TIME {
        INITIAL_DEFLATION
    } else if simulation_time < INFLATION_START_TIME + INFLATION_DURATION {
        let progress = (simulation_time - INFLATION_START_TIME) / INFLATION_DURATION;
        INITIAL_DEFLATION + (MAX_INFLATION - INITIAL_DEFLATION) * progress
    } else {
        MAX_INFLATION
    }
}

fn run_simulation(mesh_path: Option<&str>) {
    let mut mesh = load_mesh(mesh_path).expect(
        "Failed to load mesh. Make sure 'mesh/deci_spot.bin' exists or provide a valid path.",
    );
    let original_mesh = mesh.clone();
    let (mut show_wireframe, mut show_faces, mut should_reset, mut paused, mut simulation_time) =
        (true, true, false, false, 0.0);

    let (mut rl, thread) = raylib::init()
        .size(1200, 800)
        .title("XPBD Inflation Demo")
        .build();
    let (camera_pos, target) = setup_camera(Some(&mesh));
    let mut camera = Camera3D::perspective(camera_pos, target, Vector3::new(0.0, 1.0, 0.0), 60.0);
    rl.set_target_fps(TARGET_FPS.into());

    let initial_values = mesh.constraints.evaluate(&mesh.vertices);
    let mut xpbd_params = xpbd::XpbdParams {
        n_substeps: N_SUBSTEPS,
        time_substep: TIME_STEP / (N_SUBSTEPS as f32),
        length_compliance: EDGE_COMPLIANCE,
        volume_compliance: VOLUME_COMPLIANCE,
        l_threshold_length: TEARING_THRESHOLD,
        p_volume: INITIAL_DEFLATION,
        ..Default::default()
    };

    let mut state = XpbdState::new(
        mesh.vertices.len(),
        mesh.constraints.edges.len() + mesh.constraints.tetrahedra.len(),
    );
    info!("Starting inflation demo");

    while !rl.window_should_close() {
        handle_input(
            &rl,
            &mut show_wireframe,
            &mut show_faces,
            &mut should_reset,
            &mut camera,
            &mut paused,
        );
        rl.update_camera(&mut camera, CameraMode::CAMERA_THIRD_PERSON);

        if should_reset {
            mesh = original_mesh.clone();
            state = XpbdState::new(
                mesh.vertices.len(),
                mesh.constraints.edges.len() + mesh.constraints.tetrahedra.len(),
            );
            simulation_time = 0.0;
            should_reset = false;
        }

        if !paused {
            simulation_time += TIME_STEP;
            xpbd_params.p_volume = calculate_inflation_factor(simulation_time);
            state = xpbd::step_basic(&xpbd_params, state, &mut mesh, &initial_values, |v| {
                v.position.y = v.position.y.max(0.0)
            });
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::new(205, 206, 245, 255));
        {
            let mut d3 = d.begin_mode3D(camera);
            d3.draw_plane(
                Vector3::new(0.0, -0.1, 0.0),
                Vector2::new(50.0, 50.0),
                Color::new(248, 248, 255, 255),
            );
            d3.draw_plane(
                Vector3::new(0.0, 0.0, 0.0),
                Vector2::new(10.0, 10.0),
                Color::new(180, 180, 180, 255),
            );
            d3.draw_grid(20, 2.0);
            draw_mesh(&mut d3, &mesh, &state, show_wireframe, show_faces);
            d3.draw_cube_wires(Vector3::new(5.0, 0.5, 0.0), 1.0, 1.0, 1.0, Color::GRAY);
        }
        draw_ui(&mut d);
    }
}

fn main() {
    // Initialize tracing subscriber for structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    run_simulation(cli.mesh.as_deref());
}
