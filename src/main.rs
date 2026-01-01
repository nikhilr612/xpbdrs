use clap::{Parser, Subcommand};
use raylib::prelude::*;
use tracing::{debug, error, info, instrument};

use xpbdrs::{
    interaction,
    mesh::{self, Spatial, tetrahedral::MeshTearState},
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

fn handle_input(rl: &RaylibHandle, show_wireframe: &mut bool, show_faces: &mut bool, params: &mut SimParams) {
    if rl.is_key_pressed(KeyboardKey::KEY_X) {
        *show_wireframe = !*show_wireframe;
    }
    if rl.is_key_pressed(KeyboardKey::KEY_F) {
        *show_faces = !*show_faces;
    }
    // Pause/unpause simulation
    if rl.is_key_pressed(KeyboardKey::KEY_SPACE) {
        params.paused = !params.paused;
    }
    // Reset simulation
    if rl.is_key_pressed(KeyboardKey::KEY_R) {
        params.should_reset = true;
    }
    // Toggle shuffle mode (full shuffle vs no shuffle)
    if rl.is_key_pressed(KeyboardKey::KEY_T) {
        params.shuffle_buffer_size = if params.shuffle_buffer_size == usize::MAX { 1 } else { usize::MAX };
    }
    
    // Parameter adjustment step sizes
    let compliance_step = 0.00000001;
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
    
    // Interaction force: 7/8 to increase/decrease
    let force_step = 5.0;
    if rl.is_key_pressed(KeyboardKey::KEY_SEVEN) {
        params.interaction_force = (params.interaction_force + force_step).min(1000.0);
    }
    if rl.is_key_pressed(KeyboardKey::KEY_EIGHT) {
        params.interaction_force = (params.interaction_force - force_step).max(0.5);
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
    xpbd_state: &XpbdState,
    tear_state: &MeshTearState,
    show_wireframe: bool,
    show_faces: bool,
) {
    if show_faces {
        mesh.draw_faces(d3, xpbd_state, tear_state, Color::LIGHTGRAY.alpha(0.7));
    }
    if show_wireframe {
        mesh.draw_wireframe(d3, xpbd_state, tear_state, Color::BLUE);
    }
}

fn draw_ui(d: &mut RaylibDrawHandle, params: &SimParams, mesh: Option<&mesh::Tetrahedral>) {
    let screen_width = d.get_screen_width();
    
    // Left panel: Controls help
    d.draw_fps(10, 10);
    d.draw_text("=== CONTROLS ===", 10, 40, 16, Color::DARKGRAY);
    d.draw_text("X: Toggle Wireframe", 10, 60, 14, Color::MIDNIGHTBLUE);
    d.draw_text("F: Toggle Faces", 10, 78, 14, Color::MIDNIGHTBLUE);
    d.draw_text("SPACE: Pause/Resume", 10, 96, 14, Color::MIDNIGHTBLUE);
    d.draw_text("R: Reset Simulation", 10, 114, 14, Color::MIDNIGHTBLUE);
    d.draw_text("T: Toggle Shuffle", 10, 132, 14, Color::MIDNIGHTBLUE);
    d.draw_text("LMB: Push & Tear Mesh", 10, 150, 14, Color::MAROON);
    
    d.draw_text("=== ADJUST ===", 10, 176, 16, Color::DARKGRAY);
    d.draw_text("1/2: Length Compliance +/-", 10, 196, 14, Color::MIDNIGHTBLUE);
    d.draw_text("3/4: Volume Compliance +/-", 10, 214, 14, Color::MIDNIGHTBLUE);
    d.draw_text("5/6: Damping +/-", 10, 232, 14, Color::MIDNIGHTBLUE);
    d.draw_text("7/8: Interaction Force +/-", 10, 250, 14, Color::MIDNIGHTBLUE);
    d.draw_text("9/0: Gravity +/-", 10, 268, 14, Color::MIDNIGHTBLUE);
    d.draw_text("UP/DOWN: Substeps", 10, 286, 14, Color::MIDNIGHTBLUE);
    
    // Right panel: Current parameter values
    let panel_x = screen_width - 220;
    d.draw_rectangle(panel_x - 10, 30, 220, 280, Color::WHITE.alpha(0.85));
    d.draw_rectangle_lines(panel_x - 10, 30, 220, 280, Color::DARKGRAY);
    
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
    
    // Interaction parameters
    d.draw_text("--- Interaction ---", panel_x, 200, 14, Color::DARKGRAY);
    d.draw_text(
        &format!("Force:        {:.1}", params.interaction_force),
        panel_x, 218, 14, Color::MAROON
    );
    d.draw_text(
        &format!("Radius:       {:.2}", params.interaction_radius),
        panel_x, 236, 14, Color::MAROON
    );
    d.draw_text(
        &format!("Stretch:      {:.0}%", params.stretch_threshold * 100.0),
        panel_x, 254, 14, Color::MAROON
    );
    d.draw_text(
        &format!("Compress:     {:.0}%", params.compression_threshold * 100.0),
        panel_x, 272, 14, Color::MAROON
    );
    
    // Mesh info if available
    if let Some(m) = mesh {
        d.draw_text("--- Mesh ---", panel_x, 297, 14, Color::DARKGRAY);
        d.draw_text(
            &format!("Verts: {} Edges: {}", m.vertices.len(), m.constraints.edges.len()),
            panel_x, 315, 12, Color::GRAY
        );
    }
}

#[instrument]
fn load_mesh(mesh_path: &str) -> Option<mesh::Tetrahedral> {
    mesh::Tetrahedral::load_mesh(mesh_path)
        .map(|mut m| {
            m.vertices.translate(Vector3::new(0.0, 2.5, 0.0));
            m
        })
        .ok()
}

const TARGET_FPS: u16 = 60;
const TIME_STEP: f32 = 1.0 / TARGET_FPS as f32;
const N_SUBSTEPS: usize = 30;
const EDGE_COMPLIANCE: f32 = 0.00001; // Stiffer edges for stability
const VOLUME_COMPLIANCE: f32 = 0.000000001; // Stiffer volumes for stability

/// Mutable simulation state for live parameter tuning.
struct SimParams {
    length_compliance: f32,
    volume_compliance: f32,
    damping: f32,
    gravity: f32,
    n_substeps: usize,
    shuffle_buffer_size: usize,
    paused: bool,
    should_reset: bool,
    // Interaction parameters
    interaction_force: f32,
    interaction_radius: f32,
    // Tearing thresholds
    stretch_threshold: f32,    // Max stretch ratio before breaking (e.g., 2.0 = 200%)
    compression_threshold: f32, // Min compression ratio before breaking (e.g., 0.3 = 30%)
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            length_compliance: EDGE_COMPLIANCE,
            volume_compliance: VOLUME_COMPLIANCE,
            damping: 0.005, // Higher damping for stability
            gravity: -9.81,
            n_substeps: N_SUBSTEPS,
            shuffle_buffer_size: usize::MAX, // Full shuffle by default
            paused: false,
            should_reset: false,
            interaction_force: 0.2,  
            interaction_radius: 0.01,
            stretch_threshold: 1.35,      // Break when stretched to 135% of original
            compression_threshold: 0.6,  // Break when compressed to 60% of original
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
            ..Default::default()
        }
    }
    
    /// Get gravity as an acceleration field closure.
    fn gravity_field(&self) -> impl Fn(&mesh::Vertex) -> Vector3 {
        let gravity = self.gravity;
        move |_: &mesh::Vertex| Vector3::new(0.0, gravity, 0.0)
    }
}

#[instrument]
fn run_simulation(mesh_path: Option<&str>) {
    let mut mesh = mesh_path.and_then(load_mesh);
    let original_mesh = mesh.clone();
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
    let mut tear_state = mesh.as_ref().map(|m| {
        MeshTearState::with_adjacency(&m.faces, &m.constraints.edges)
    });

    // Track active interaction ray for visualization
    let mut active_ray: Option<interaction::CylindricalRay>;

    while !rl.window_should_close() {
        handle_input(&rl, &mut show_wireframe, &mut show_faces, &mut sim_params);
        rl.update_camera(&mut camera, CameraMode::CAMERA_THIRD_PERSON);

        // Reset simulation if requested
        if sim_params.should_reset {
            if let Some(original) = &original_mesh {
                mesh = Some(original.clone());
                if let Some(m) = &mesh {
                    state = Some(XpbdState::new(
                        m.vertices.len(),
                        m.constraints.edges.len() + m.constraints.tetrahedra.len(),
                    ));
                    tear_state = Some(MeshTearState::with_adjacency(&m.faces, &m.constraints.edges));
                }
            }
            sim_params.should_reset = false;
        }

        // Handle mouse interaction
        if rl.is_mouse_button_down(MouseButton::MOUSE_BUTTON_LEFT) {
            let mouse_x = rl.get_mouse_x();
            let mouse_y = rl.get_mouse_y();
            let screen_width = rl.get_screen_width();
            let screen_height = rl.get_screen_height();

            // Create cylindrical ray from camera through mouse position
            let cyl_ray = interaction::CylindricalRay::from_camera_mouse(
                &camera,
                screen_width,
                screen_height,
                mouse_x,
                mouse_y,
                sim_params.interaction_radius,
            );

            // Store for visualization
            active_ray = Some(cyl_ray);

            if let (Some(mesh), Some(st), Some(ray)) = (
                &mut mesh,
                &mut state,
                &active_ray,
            ) {
                // Compute interaction effect
                let effect = interaction::InteractionEffect::from_cylindrical_ray(
                    ray,
                    &mesh.vertices,
                    sim_params.interaction_force,
                );

                // Apply forces through the solver's force interface (proper physics integration)
                st.apply_interaction_effect(&effect);
            }
        } else {
            active_ray = None;
        }

        // Only step simulation if not paused
        if !sim_params.paused {
            if let (Some(mesh), Some(st), Some(ts), Some(initial_vals)) = (
                &mut mesh,
                &mut state,
                &mut tear_state,
                &initial_values,
            ) {
                let current_state = st.clone();
                let xpbd_params = sim_params.to_xpbd_params();
                let gravity_field = sim_params.gravity_field();
                *st = xpbd::step_basic(
                    &xpbd_params,
                    current_state,
                    mesh,
                    initial_vals,
                    |v| v.position.y = v.position.y.max(0.0), // ground at y=0
                    gravity_field,
                );
                
                // Check and break overstretched/compressed edges every frame
                mesh.tear_edges(
                    st,
                    ts,
                    initial_vals,
                    sim_params.stretch_threshold,
                    sim_params.compression_threshold,
                );
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
            if let (Some(mesh), Some(st), Some(ts)) = (&mesh, &state, &tear_state) {
                draw_mesh(&mut d3, mesh, st, ts, show_wireframe, show_faces);
            }

            // Draw interaction ray cylinder when active
            if let Some(ray) = &active_ray {
                let ray_length = 50.0; // Draw ray extending into scene
                
                // Find perpendicular vectors to the ray direction
                let up = if ray.ray.direction.y.abs() < 0.9 {
                    Vector3::new(0.0, 1.0, 0.0)
                } else {
                    Vector3::new(1.0, 0.0, 0.0)
                };
                let right = ray.ray.direction.cross(up).normalized();
                let actual_up = right.cross(ray.ray.direction).normalized();
                
                // Draw multiple lines along the cylinder surface
                let num_lines = 8;
                for i in 0..num_lines {
                    let angle = (i as f32 / num_lines as f32) * std::f32::consts::TAU;
                    let offset = (right * angle.cos() + actual_up * angle.sin()) * ray.radius;
                    
                    let start = ray.ray.origin + offset;
                    let end = ray.ray.origin + ray.ray.direction * ray_length + offset;
                    
                    d3.draw_line_3D(start, end, Color::RED);
                }
                
                // Draw center line of the ray (thicker by drawing multiple times)
                let end_point = ray.ray.origin + ray.ray.direction * ray_length;
                d3.draw_line_3D(ray.ray.origin, end_point, Color::YELLOW);
                
                // Draw circles at start and a few points along
                let ring_segments = 16;
                for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let center = ray.ray.origin + ray.ray.direction * (ray_length * t);
                    for j in 0..ring_segments {
                        let angle1 = (j as f32 / ring_segments as f32) * std::f32::consts::TAU;
                        let angle2 = ((j + 1) as f32 / ring_segments as f32) * std::f32::consts::TAU;
                        
                        let p1 = center + (right * angle1.cos() + actual_up * angle1.sin()) * ray.radius;
                        let p2 = center + (right * angle2.cos() + actual_up * angle2.sin()) * ray.radius;
                        
                        d3.draw_line_3D(p1, p2, Color::RED);
                    }
                }
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
