use clap::{Parser, Subcommand};
use raylib::prelude::*;
use tracing::{debug, error, info, instrument};

use xpbdrs::{
    mesh::{self, Spatial},
    xpbd::{self, ConstraintSet, XpbdState},
};

#[derive(Parser)]
#[command(name = "xpbdcloth")]
#[command(about = "Extended Position Based Dynamics cloth simulation")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Export tetgen files to binary format
    Export {
        /// Input file prefix (without extension)
        #[arg(short, long)]
        input: String,
        /// Output binary file path
        #[arg(short, long)]
        output: String,
    },
    /// Run the simulation with a demo mesh.
    Demo {
        /// Optional mesh file prefix to visualize
        mesh: Option<String>,
    },
}

#[instrument]
fn export_mesh(input_prefix: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!(input_prefix, "Loading tetrahedral mesh");

    let mesh = mesh::Tetrahedral::from_files(input_prefix)?;

    info!(
        vertices = mesh.vertices.len(),
        edges = mesh.constraints.edges.len(),
        faces = mesh.faces.len(),
        tetrahedra = mesh.constraints.tetrahedra.len(),
        "Mesh loaded successfully"
    );

    mesh.export_to_bincode(output_path)?;

    Ok(())
}

#[instrument(skip(mesh))]
fn setup_camera(mesh: Option<&mesh::Tetrahedral>) -> (Vector3, Vector3) {
    mesh.map_or_else(
        || (Vector3::new(7.0, 7.0, 7.0), Vector3::new(0.0, 0.0, 0.0)),
        |mesh| {
            let (min, max) = mesh.vertices.bounding_box();
            debug!(
                min_x = %min.x, min_y = %min.y, min_z = %min.z,
                max_x = %max.x, max_y = %max.y, max_z = %max.z,
                "Mesh bounding box"
            );

            let center = Vector3::new(
                (min.x + max.x) * 0.5,
                (min.y + max.y) * 0.5,
                (min.z + max.z) * 0.5,
            );
            let size = (max - min).length().max(1.0); // Ensure minimum size
            let distance = size * 2.5;

            debug!(
                center_x = %center.x, center_y = %center.y, center_z = %center.z,
                size = %size,
                "Camera setup - mesh center and size"
            );

            // Position camera at 45-degree angle for good visibility
            let camera_pos = Vector3::new(
                center.x + distance * 0.7,
                center.y + distance * 0.7,
                center.z + distance * 0.7,
            );

            debug!(
                camera_x = %camera_pos.x, camera_y = %camera_pos.y, camera_z = %camera_pos.z,
                "Camera position calculated"
            );

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

    // Camera Y-axis movement
    const CAMERA_SPEED: f32 = 0.1;
    if rl.is_key_down(KeyboardKey::KEY_SPACE) {
        // Space: Move camera up
        camera.position.y += CAMERA_SPEED;
        camera.target.y += CAMERA_SPEED;
    } else if rl.is_key_down(KeyboardKey::KEY_LEFT_SHIFT)
        || rl.is_key_down(KeyboardKey::KEY_RIGHT_SHIFT)
    {
        // Shift: Move camera down
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
        mesh.draw_faces(d3, state, Color::new(255, 215, 0, 200)); // Gold with transparency
    }
    if show_wireframe {
        mesh.draw_wireframe(d3, Color::new(30, 144, 255, 255)); // Dodger blue
    }
}

fn draw_ui(d: &mut RaylibDrawHandle) {
    d.draw_fps(10, 10);
    d.draw_text("X: Wireframe", 10, 40, 15, Color::BLACK);
    d.draw_text("F: Faces", 10, 60, 15, Color::BLACK);
    d.draw_text("R: Reset", 10, 80, 15, Color::BLACK);
    d.draw_text("Space: Camera Up", 10, 100, 15, Color::BLACK);
    d.draw_text("Shift: Camera Down", 10, 120, 15, Color::BLACK);
}

#[instrument]
fn load_mesh(mesh_path: &str) -> Option<mesh::Tetrahedral> {
    let load_result = if std::path::Path::new(mesh_path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
    {
        mesh::Tetrahedral::from_bincode(mesh_path)
    } else {
        mesh::Tetrahedral::from_files(mesh_path)
    };

    load_result
        .map(|mut m| {
            m.vertices.translate(Vector3::new(0.0, 2.5, 0.0));
            m
        })
        .ok()
}

const TARGET_FPS: u16 = 60;
const TIME_STEP: f32 = 1.0 / TARGET_FPS as f32;
const N_SUBSTEPS: usize = 10;
const EDGE_COMPLIANCE: f32 = 0.00;
const VOLUME_COMPLIANCE: f32 = 0.00;

#[instrument]
fn run_simulation(mesh_path: Option<&str>) {
    let mut mesh = mesh_path.and_then(load_mesh);
    let original_mesh = mesh.clone();
    let mut show_wireframe = true;
    let mut show_faces = false;
    let mut should_reset = false;

    let (mut rl, thread) = raylib::init()
        .size(800, 600)
        .title("XPBD Cloth Simulation")
        .build();

    let (camera_pos, target) = setup_camera(mesh.as_ref());
    let mut camera = Camera3D::perspective(camera_pos, target, Vector3::new(0.0, 1.0, 0.0), 60.0);
    rl.set_target_fps(TARGET_FPS.into());

    let initial_values = mesh.as_ref().map(|m| m.constraints.evaluate(&m.vertices));
    let xpbd_params = xpbd::XpbdParams {
        n_substeps: N_SUBSTEPS,
        time_substep: TIME_STEP / (N_SUBSTEPS as f32),
        length_compliance: EDGE_COMPLIANCE,
        volume_compliance: VOLUME_COMPLIANCE,
        ..Default::default()
    };
    let mut state = mesh.as_ref().map(|m| {
        XpbdState::new(
            m.vertices.len(),
            m.constraints.edges.len() + m.constraints.tetrahedra.len(),
        )
    });

    while !rl.window_should_close() {
        handle_input(
            &rl,
            &mut show_wireframe,
            &mut show_faces,
            &mut should_reset,
            &mut camera,
        );
        rl.update_camera(&mut camera, CameraMode::CAMERA_THIRD_PERSON);

        // Reset simulation if space was pressed
        if should_reset {
            if let Some(original) = &original_mesh {
                mesh = Some(original.clone());
                if let Some(mesh) = &mesh {
                    state = Some(XpbdState::new(
                        mesh.vertices.len(),
                        mesh.constraints.edges.len() + mesh.constraints.tetrahedra.len(),
                    ));
                }
            }
            should_reset = false;
        }

        if let Some(mesh) = &mut mesh {
            let current_state = state.take().unwrap();
            state = Some(xpbd::step_basic(
                &xpbd_params,
                current_state,
                mesh,
                initial_values.as_ref().unwrap(),
                |v| v.position.y = v.position.y.max(0.0), // ground at y=0
            ));
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::new(205, 206, 245, 255)); // Light blue sky

        {
            let mut d3 = d.begin_mode3D(camera);

            // Draw large white floor plane
            d3.draw_plane(
                Vector3::new(0.0, -0.1, 0.0),
                Vector2::new(50.0, 50.0),
                Color::new(248, 248, 255, 255), // Ghost white
            );

            // Always draw ground plane and grid with better contrast
            d3.draw_plane(
                Vector3::new(0.0, 0.0, 0.0),
                Vector2::new(10.0, 10.0),
                Color::new(180, 180, 180, 255), // Light gray
            );
            d3.draw_grid(20, 2.0);

            // Draw mesh if loaded
            if let Some(mesh) = &mesh {
                draw_mesh(
                    &mut d3,
                    mesh,
                    state.as_ref().unwrap(),
                    show_wireframe,
                    show_faces,
                );
            }
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

    match cli.command {
        Commands::Export { input, output } => {
            if let Err(e) = export_mesh(&input, &output) {
                error!(error = %e, "Export failed");
                std::process::exit(1);
            }
        }
        Commands::Demo { mesh } => {
            run_simulation(mesh.as_deref());
        }
    }
}
