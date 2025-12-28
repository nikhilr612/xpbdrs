use clap::{Parser, Subcommand};
use raylib::prelude::*;
use tracing::{debug, error, info, instrument};

use xpbdrs::{
    mesh::{self, Mesh},
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
            let (min, max) = mesh.bounding_box();
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

fn handle_input(rl: &RaylibHandle, show_wireframe: &mut bool, show_faces: &mut bool, params: &mut SimParams) {
    if rl.is_key_pressed(KeyboardKey::KEY_R) {
        *show_wireframe = !*show_wireframe;
    }
    if rl.is_key_pressed(KeyboardKey::KEY_F) {
        *show_faces = !*show_faces;
    }
    // Pause/unpause simulation
    if rl.is_key_pressed(KeyboardKey::KEY_SPACE) {
        params.paused = !params.paused;
    }
    // Toggle shuffle mode (full shuffle vs no shuffle)
    if rl.is_key_pressed(KeyboardKey::KEY_T) {
        params.shuffle_buffer_size = if params.shuffle_buffer_size == usize::MAX { 1 } else { usize::MAX };
    }
    
    // Parameter adjustment step sizes
    let compliance_step = 0.0000001;
    let damping_step = 0.0005;
    let gravity_step = 0.1;
    
    // Edge compliance: 1/2 to increase/decrease
    if rl.is_key_pressed(KeyboardKey::KEY_ONE) {
        params.length_compliance = (params.length_compliance + compliance_step).min(0.001);
    }
    if rl.is_key_pressed(KeyboardKey::KEY_TWO) {
        params.length_compliance = (params.length_compliance - compliance_step).max(0.0);
    }
    
    // Volume compliance: 3/4 to increase/decrease
    if rl.is_key_pressed(KeyboardKey::KEY_THREE) {
        params.volume_compliance = (params.volume_compliance + compliance_step).min(0.001);
    }
    if rl.is_key_pressed(KeyboardKey::KEY_FOUR) {
        params.volume_compliance = (params.volume_compliance - compliance_step).max(0.0);
    }
    
    // Damping: 5/6 to increase/decrease
    if rl.is_key_pressed(KeyboardKey::KEY_FIVE) {
        params.damping = (params.damping + damping_step).min(1.0);
    }
    if rl.is_key_pressed(KeyboardKey::KEY_SIX) {
        params.damping = (params.damping - damping_step).max(0.0);
    }
    
    // Gravity: 9/0 to increase/decrease magnitude
    if rl.is_key_pressed(KeyboardKey::KEY_NINE) {
        params.gravity -= gravity_step; // More negative = stronger gravity
    }
    if rl.is_key_pressed(KeyboardKey::KEY_ZERO) {
        params.gravity += gravity_step;
    }
    
    // Substeps: UP/DOWN to adjust
    if rl.is_key_pressed(KeyboardKey::KEY_UP) {
        params.n_substeps = (params.n_substeps + 5).min(100);
    }
    if rl.is_key_pressed(KeyboardKey::KEY_DOWN) {
        params.n_substeps = params.n_substeps.saturating_sub(5).max(1);
    }
}

fn draw_mesh(
    d3: &mut RaylibMode3D<RaylibDrawHandle>,
    mesh: &mesh::Tetrahedral,
    show_wireframe: bool,
    show_faces: bool,
) {
    if show_faces {
        mesh.draw_faces(d3, Color::LIGHTGRAY.alpha(0.7));
    }
    if show_wireframe {
        mesh.draw_wireframe(d3, Color::BLUE);
    }
}

fn draw_ui(d: &mut RaylibDrawHandle, params: &SimParams, mesh: Option<&mesh::Tetrahedral>) {
    let screen_width = d.get_screen_width();
    
    // Left panel: Controls help
    d.draw_fps(10, 10);
    d.draw_text("=== CONTROLS ===", 10, 40, 16, Color::DARKGRAY);
    d.draw_text("R: Toggle Wireframe", 10, 60, 14, Color::MIDNIGHTBLUE);
    d.draw_text("F: Toggle Faces", 10, 78, 14, Color::MIDNIGHTBLUE);
    d.draw_text("SPACE: Pause/Resume", 10, 96, 14, Color::MIDNIGHTBLUE);
    d.draw_text("T: Toggle Shuffle", 10, 114, 14, Color::MIDNIGHTBLUE);
    
    d.draw_text("=== ADJUST ===", 10, 140, 16, Color::DARKGRAY);
    d.draw_text("1/2: Length Compliance +/-", 10, 160, 14, Color::MIDNIGHTBLUE);
    d.draw_text("3/4: Volume Compliance +/-", 10, 178, 14, Color::MIDNIGHTBLUE);
    d.draw_text("5/6: Damping +/-", 10, 196, 14, Color::MIDNIGHTBLUE);
    d.draw_text("9/0: Gravity +/-", 10, 214, 14, Color::MIDNIGHTBLUE);
    d.draw_text("UP/DOWN: Substeps", 10, 232, 14, Color::MIDNIGHTBLUE);
    
    // Right panel: Current parameter values
    let panel_x = screen_width - 220;
    d.draw_rectangle(panel_x - 10, 30, 220, 220, Color::WHITE.alpha(0.85));
    d.draw_rectangle_lines(panel_x - 10, 30, 220, 220, Color::DARKGRAY);
    
    d.draw_text("=== PARAMETERS ===", panel_x, 40, 16, Color::DARKGRAY);
    
    // Status indicator
    let status_text = if params.paused { "PAUSED" } else { "RUNNING" };
    let status_color = if params.paused { Color::RED } else { Color::GREEN };
    d.draw_text(status_text, panel_x, 60, 18, status_color);
    
    // Parameter values
    d.draw_text(
        &format!("Length Compl: {:.2e}", params.length_compliance),
        panel_x, 85, 14, Color::DARKBLUE
    );
    d.draw_text(
        &format!("Vol Compl:    {:.2e}", params.volume_compliance),
        panel_x, 103, 14, Color::DARKBLUE
    );
    d.draw_text(
        &format!("Damping:      {:.5}", params.damping),
        panel_x, 121, 14, Color::DARKBLUE
    );
    d.draw_text(
        &format!("Gravity:      {:.2}", params.gravity),
        panel_x, 139, 14, Color::DARKBLUE
    );
    d.draw_text(
        &format!("Substeps:     {}", params.n_substeps),
        panel_x, 157, 14, Color::DARKBLUE
    );
    
    let shuffle_text = if params.shuffle_buffer_size == usize::MAX { "FULL" } else { "OFF" };
    let shuffle_color = if params.shuffle_buffer_size == usize::MAX { Color::GREEN } else { Color::GRAY };
    d.draw_text(&format!("Shuffle:      {}", shuffle_text), panel_x, 175, 14, shuffle_color);
    
    // Mesh info if available
    if let Some(m) = mesh {
        d.draw_text("--- Mesh ---", panel_x, 200, 14, Color::DARKGRAY);
        d.draw_text(
            &format!("Verts: {} Edges: {}", m.vertices.len(), m.constraints.edges.len()),
            panel_x, 218, 12, Color::GRAY
        );
    }
}

#[instrument]
fn load_mesh(mesh_path: &str) -> Option<mesh::Tetrahedral> {
    mesh::Tetrahedral::load_mesh(mesh_path)
        .map(|mut m| {
            m.translate(Vector3::new(0.0, 2.5, 0.0));
            m
        })
        .ok()
}

const TARGET_FPS: u16 = 60;
const TIME_STEP: f32 = 1.0 / TARGET_FPS as f32;
const N_SUBSTEPS: usize = 30;
const EDGE_COMPLIANCE: f32 = 0.0000008;
const VOLUME_COMPLIANCE: f32 = 0.00;

/// Mutable simulation state for live parameter tuning.
struct SimParams {
    length_compliance: f32,
    volume_compliance: f32,
    damping: f32,
    gravity: f32,
    n_substeps: usize,
    shuffle_buffer_size: usize,
    paused: bool,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            length_compliance: EDGE_COMPLIANCE,
            volume_compliance: VOLUME_COMPLIANCE,
            damping: 0.0005,
            gravity: -9.81,
            n_substeps: N_SUBSTEPS,
            shuffle_buffer_size: usize::MAX, // Full shuffle by default
            paused: false,
        }
    }
}

impl SimParams {
    /// Convert to XpbdParams for the simulation.
    fn to_xpbd_params(&self) -> xpbd::XpbdParams {
        xpbd::XpbdParams {
            n_substeps: self.n_substeps,
            time_substep: TIME_STEP / (self.n_substeps as f32),
            length_compliance: self.length_compliance,
            volume_compliance: self.volume_compliance,
            damping: self.damping,
            shuffle_buffer_size: self.shuffle_buffer_size,
            constant_field: Vector3::new(0.0, self.gravity, 0.0),
            ..Default::default()
        }
    }
}

#[instrument]
fn run_simulation(mesh_path: Option<&str>) {
    let mut mesh = mesh_path.and_then(load_mesh);
    let mut show_wireframe = true;
    let mut show_faces = false;
    let mut sim_params = SimParams::default();

    let (mut rl, thread) = raylib::init()
        .size(1000, 1000)
        .title("XPBD Cloth Simulation")
        .build();

    let (camera_pos, target) = setup_camera(mesh.as_ref());
    let mut camera = Camera3D::perspective(camera_pos, target, Vector3::new(0.0, 1.0, 0.0), 60.0);
    rl.set_target_fps(TARGET_FPS.into());

    let initial_values = mesh.as_ref().map(|m| m.constraints.evaluate(&m.vertices));
    let mut state = mesh.as_ref().map(|m| {
        XpbdState::new(
            m.vertices.len(),
            m.constraints.edges.len() + m.constraints.tetrahedra.len(),
        )
    });

    while !rl.window_should_close() {
        handle_input(&rl, &mut show_wireframe, &mut show_faces, &mut sim_params);
        rl.update_camera(&mut camera, CameraMode::CAMERA_THIRD_PERSON);

        // Only step simulation if not paused
        if !sim_params.paused {
            if let Some(mesh) = &mut mesh {
                let current_state = state.take().unwrap();
                let xpbd_params = sim_params.to_xpbd_params();
                state = Some(xpbd::step_basic(
                    &xpbd_params,
                    current_state,
                    mesh,
                    initial_values.as_ref().unwrap(),
                    |v| v.position.y = v.position.y.max(0.0), // ground at y=0
                ));
            }
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::RAYWHITE);

        {
            let mut d3 = d.begin_mode3D(camera);

            // Always draw ground plane and grid
            d3.draw_plane(
                Vector3::new(0.0, 0.0, 0.0),
                Vector2::new(10.0, 10.0),
                Color::GRAY,
            );
            d3.draw_grid(20, 2.0);

            // Draw mesh if loaded
            if let Some(mesh) = &mesh {
                draw_mesh(&mut d3, mesh, show_wireframe, show_faces);
            }
        }

        draw_ui(&mut d, &sim_params, mesh.as_ref());
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
