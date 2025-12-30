//! Spot in a box demo - spawns deci_spot mesh with gravity and box collisions,
//! applying preset forces periodically at the centroid.

use raylib::prelude::*;
use xpbdrs::{
    mesh::{self, Spatial},
    xpbd::{self, ConstraintSet, XpbdState},
};

const TARGET_FPS: u16 = 60;
const TIME_STEP: f32 = 1.0 / TARGET_FPS as f32;
const N_SUBSTEPS: usize = 10;
const EDGE_COMPLIANCE: f32 = 0.00;
const VOLUME_COMPLIANCE: f32 = 0.00;

const BOX_SIZE: f32 = 2.0;
const FORCE_MAGNITUDE: f32 = 20.0; // Comparable to gravity
const FORCE_THRESHOLD: f32 = 0.5; // Minimum distance threshold
const MAX_FORCE: f32 = 80.0; // Force cap
const FORCE_DURATION: f32 = 0.5; // Duration to apply force

// Preset force directions
const FORCES: &[Vector3] = &[
    Vector3::new(1.0, 0.5, 0.0),
    Vector3::new(-1.0, 0.8, 0.0),
    Vector3::new(0.0, 1.0, 1.0),
    Vector3::new(0.0, 0.5, -1.0),
];

fn calculate_centroid(mesh: &mesh::Tetrahedral) -> Vector3 {
    let mut centroid = Vector3::zero();
    for vertex in mesh.vertices.iter() {
        centroid += vertex.position;
    }
    centroid / mesh.vertices.len() as f32
}

fn main() {
    let mut mesh = mesh::Tetrahedral::from_bincode("mesh/deci_spot.bin")
        .expect("Failed to load mesh/deci_spot.bin");
    mesh.vertices.translate(Vector3::new(0.0, 2.0, 0.0));

    let original_mesh = mesh.clone();
    let mut time = 0.0;
    let mut force_index = 0;

    let (mut rl, thread) = raylib::init().size(800, 800).title("Spot in a Box").build();
    let mut camera = Camera3D::perspective(
        Vector3::new(7.0, 7.0, 7.0),
        Vector3::new(0.0, 2.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        60.0,
    );
    rl.set_target_fps(TARGET_FPS.into());

    let initial_values = mesh.constraints.evaluate(&mesh.vertices);
    let xpbd_params = xpbd::XpbdParams {
        n_substeps: N_SUBSTEPS,
        time_substep: TIME_STEP / N_SUBSTEPS as f32,
        length_compliance: EDGE_COMPLIANCE,
        volume_compliance: VOLUME_COMPLIANCE,
        ..Default::default()
    };

    let mut state = XpbdState::new(
        mesh.vertices.len(),
        mesh.constraints.edges.len() + mesh.constraints.tetrahedra.len(),
    );

    while !rl.window_should_close() {
        if rl.is_key_pressed(KeyboardKey::KEY_R) {
            mesh = original_mesh.clone();
            state = XpbdState::new(
                mesh.vertices.len(),
                mesh.constraints.edges.len() + mesh.constraints.tetrahedra.len(),
            );
            time = 0.0;
            force_index = 0;
        }

        rl.update_camera(&mut camera, CameraMode::CAMERA_THIRD_PERSON);
        time += TIME_STEP;

        // Apply force every 1.5 seconds for 0.1 seconds
        let force_active = (time % 1.5) < FORCE_DURATION;
        if force_active && (time % 1.5) < TIME_STEP {
            force_index = (force_index + 1) % FORCES.len();
        }

        let centroid = calculate_centroid(&mesh);
        let current_force = if force_active {
            FORCES[force_index].normalized()
        } else {
            Vector3::zero()
        };

        // Box collision
        let mut box_collision = |vertex: &mut mesh::Vertex| {
            vertex.position.x = vertex.position.x.clamp(-BOX_SIZE, BOX_SIZE);
            vertex.position.y = vertex.position.y.clamp(0.0, BOX_SIZE * 2.0);
            vertex.position.z = vertex.position.z.clamp(-BOX_SIZE, BOX_SIZE);
        };

        // Physics with inverse square law
        let acceleration_field = |vertex: &mesh::Vertex| {
            let mut acceleration = Vector3::new(0.0, -9.81, 0.0); // Gravity
            if force_active {
                let distance = (vertex.position - centroid).length();
                if distance >= FORCE_THRESHOLD {
                    let force_mag = (FORCE_MAGNITUDE / (distance * distance)).min(MAX_FORCE);
                    acceleration += current_force * force_mag * vertex.inv_mass;
                }
            }
            acceleration
        };

        for _ in 0..xpbd_params.n_substeps {
            xpbd::substep(
                &xpbd_params,
                &mut state,
                &mut mesh.vertices,
                &mesh.constraints,
                &initial_values,
                &mut box_collision,
                &acceleration_field,
            );
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::new(205, 206, 245, 255));

        {
            let mut d3 = d.begin_mode3D(camera);
            d3.draw_grid(20, 1.0);

            // Box wireframe
            d3.draw_cube_wires(
                Vector3::new(0.0, BOX_SIZE, 0.0),
                BOX_SIZE * 2.0,
                BOX_SIZE * 2.0,
                BOX_SIZE * 2.0,
                Color::RED,
            );

            // Mesh
            mesh.draw_faces(&mut d3, &state, Color::new(255, 200, 100, 200));
            mesh.draw_wireframe(&mut d3, Color::new(30, 144, 255, 255));

            // Centroid when force is active
            if force_active {
                d3.draw_sphere(centroid, 0.08, Color::RED);
            }
        }
    }
}
