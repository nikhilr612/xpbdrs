//! Drawing utilities for visualizing Xpbd simulations using Raylib.
//! Requires the `raylib` feature to be enabled.

use glam::Vec3;
use raylib::{
    color::Color,
    math::Vector3,
    prelude::{RaylibDraw3D, RaylibDrawHandle, RaylibMode3D},
};

use crate::{
    mesh::{Tetrahedral, TriangulatedSurface},
    xpbd::XpbdState,
};

impl Tetrahedral {
    /// Draw wireframe of the mesh.
    pub fn draw_wireframe(&self, d3: &mut RaylibMode3D<RaylibDrawHandle>, color: Color) {
        // Draw explicit edges if available
        for edge in &self.constraints.edges {
            if let (Some(v1), Some(v2)) = (
                self.vertices.get((edge.0.0 - 1) as usize),
                self.vertices.get((edge.1.0 - 1) as usize),
            ) {
                let start = to_ray_vector3(v1.position);
                let end = to_ray_vector3(v2.position);
                d3.draw_line_3D(start, end, color);
            }
        }
    }

    /// Draw filled faces.
    pub fn draw_faces(
        &self,
        d3: &mut RaylibMode3D<RaylibDrawHandle>,
        state: &XpbdState,
        color: Color,
    ) {
        for face in &self.faces {
            let verts = [
                self.vertices[(face.verts[0].0 - 1) as usize],
                self.vertices[(face.verts[1].0 - 1) as usize],
                self.vertices[(face.verts[2].0 - 1) as usize],
            ];

            // A triangle is "torn" if any of its corresponding edge constraints are inactive.
            let torn = face
                .edges
                .iter()
                .filter_map(|e| e.as_ref()) // Only check edges that have constraints
                .any(|e| state.constraint_inactive(e.0 as usize)); // in this constraint set, edges are solved first, so base index is 0.

            if !torn {
                d3.draw_triangle3D(
                    to_ray_vector3(verts[0].position),
                    to_ray_vector3(verts[1].position),
                    to_ray_vector3(verts[2].position),
                    color,
                );
            }
        }
    }
}

fn to_ray_vector3(v: Vec3) -> Vector3 {
    Vector3::new(v.x, v.y, v.z)
}

impl TriangulatedSurface {
    /// Draw wireframe of the mesh.
    pub fn draw_wireframe(&self, d3: &mut RaylibMode3D<RaylibDrawHandle>, color: Color) {
        // Draw explicit edges if available
        for edge in &self.constraints.edges {
            if let (Some(v1), Some(v2)) = (
                self.vertices.get((edge.0.0 - 1) as usize),
                self.vertices.get((edge.1.0 - 1) as usize),
            ) {
                let start = to_ray_vector3(v1.position);
                let end = to_ray_vector3(v2.position);
                d3.draw_line_3D(start, end, color);
            }
        }
    }

    /// Draw weak bending constraints.
    pub fn draw_weak_bending(&self, d3: &mut RaylibMode3D<RaylibDrawHandle>, color: Color) {
        for edge in &self.constraints.weak_bending {
            if let (Some(v1), Some(v2)) = (
                self.vertices.get((edge.0.0 - 1) as usize),
                self.vertices.get((edge.1.0 - 1) as usize),
            ) {
                let start = to_ray_vector3(v1.position);
                let end = to_ray_vector3(v2.position);
                d3.draw_line_3D(start, end, color);
            }
        }
    }

    /// Draw filled faces.
    /// # Panics
    /// Panics if any face does not have corresponding edges.
    pub fn draw_faces(
        &self,
        d3: &mut RaylibMode3D<RaylibDrawHandle>,
        state: &XpbdState,
        color: Color,
    ) {
        for face in &self.faces {
            let verts = [
                self.vertices[(face.verts[0].0 - 1) as usize],
                self.vertices[(face.verts[1].0 - 1) as usize],
                self.vertices[(face.verts[2].0 - 1) as usize],
            ];

            // A triangle is "torn" if any of its corresponding edge constraints are inactive.
            let torn = face.edges.iter().any(|e| {
                state.constraint_inactive(
                    e.expect("All triangles should have corresponding edges").0 as usize,
                )
            }); // edges are solved first, so base index is 0.

            if !torn {
                d3.draw_triangle3D(
                    to_ray_vector3(verts[0].position),
                    to_ray_vector3(verts[1].position),
                    to_ray_vector3(verts[2].position),
                    color,
                );
            }
        }
    }
}
