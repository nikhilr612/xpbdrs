//! Ray casting and mesh interaction system.

use raylib::prelude::*;

/// A ray cast from a point in a direction.
#[derive(Clone, Debug)]
pub struct Ray {
    /// Origin point of the ray.
    pub origin: Vector3,
    /// Normalized direction of the ray.
    pub direction: Vector3,
}

/// Represents a cylindrical interaction volume around a ray.
pub struct CylindricalRay {
    /// The central ray of the cylinder.
    pub ray: Ray,
    /// The radius of the cylinder around the ray.
    pub radius: f32,
}

impl CylindricalRay {
    /// Create a new cylindrical ray.
    pub fn new(origin: Vector3, direction: Vector3, radius: f32) -> Self {
        let normalized_direction = direction.normalized();
        Self {
            ray: Ray {
                origin,
                direction: normalized_direction,
            },
            radius,
        }
    }

    /// Check if a point is within the cylindrical ray volume.
    ///
    /// Returns (is_inside, distance_along_ray).
    pub fn point_intersection(&self, point: Vector3) -> (bool, f32) {
        let to_point = point - self.ray.origin;
        let distance_along_ray = to_point.dot(self.ray.direction);

        if distance_along_ray < 0.0 {
            // Point is behind the ray origin
            return (false, distance_along_ray);
        }

        // Project point onto ray
        let closest_point = self.ray.origin + self.ray.direction * distance_along_ray;
        let distance_to_ray = (point - closest_point).length();

        (distance_to_ray <= self.radius, distance_along_ray)
    }

    /// Cast a ray from camera through a mouse position on the screen.
    ///
    /// Uses the camera's perspective to create a ray that passes through the mouse coords.
    pub fn from_camera_mouse(
        camera: &Camera3D,
        screen_width: i32,
        screen_height: i32,
        mouse_x: i32,
        mouse_y: i32,
        radius: f32,
    ) -> Self {
        // Normalize mouse coordinates to NDC [-1, 1]
        let ndc_x = (2.0 * mouse_x as f32) / screen_width as f32 - 1.0;
        let ndc_y = 1.0 - (2.0 * mouse_y as f32) / screen_height as f32;

        // For perspective camera, compute ray direction
        let camera_forward = (camera.target - camera.position).normalized();
        let camera_right = camera_forward.cross(camera.up).normalized();
        let camera_up = camera_right.cross(camera_forward).normalized();

        // Field of view (in degrees) - we'll use a reasonable default
        let fov_y = 60.0 * std::f32::consts::PI / 180.0;
        let aspect = screen_width as f32 / screen_height as f32;

        // Ray direction based on perspective projection
        let half_height = (fov_y / 2.0).tan();
        let half_width = half_height * aspect;

        let ray_direction =
            (camera_forward + camera_right * (ndc_x * half_width) + camera_up * (ndc_y * half_height))
                .normalized();

        CylindricalRay::new(camera.position, ray_direction, radius)
    }
}

/// Represents the effect of an interaction on the mesh.
pub struct InteractionEffect {
    /// Indices of affected vertices.
    pub affected_vertices: Vec<usize>,
    /// Force magnitude to apply to each vertex (distance-based falloff).
    pub forces: Vec<Vector3>,
}

impl InteractionEffect {
    /// Create empty interaction effect.
    pub fn new() -> Self {
        Self {
            affected_vertices: Vec::new(),
            forces: Vec::new(),
        }
    }

    /// Compute interaction effect for vertices hit by the cylindrical ray.
    ///
    /// Applies a push force along the ray direction, with inverse square law falloff.
    pub fn from_cylindrical_ray(
        ray: &CylindricalRay,
        vertices: &[crate::mesh::Vertex],
        force_magnitude: f32,
    ) -> Self {
        let mut effect = InteractionEffect::new();

        for (idx, vertex) in vertices.iter().enumerate() {
            let (is_inside, distance) = ray.point_intersection(vertex.position);

            if is_inside && distance >= 0.0 {
                // Compute distance from ray axis
                let to_point = vertex.position - ray.ray.origin;
                let distance_to_ray_axis = (to_point - ray.ray.direction * to_point.dot(ray.ray.direction)).length();
                
                // Inverse square law falloff: F = k / (r^2 + epsilon)
                // Add small epsilon to avoid division by zero at the center
                let epsilon = 0.000001;
                let falloff = 1.0 / (distance_to_ray_axis * distance_to_ray_axis + epsilon);

                // Force along ray direction with inverse square falloff
                let force = ray.ray.direction * (force_magnitude * falloff);

                effect.affected_vertices.push(idx);
                effect.forces.push(force);
            }
        }

        effect
    }
}

impl Default for InteractionEffect {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_creation() {
        let ray = Ray {
            origin: Vector3::new(0.0, 0.0, 0.0),
            direction: Vector3::new(1.0, 0.0, 0.0),
        };
        assert!((ray.direction.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cylindrical_ray_point_intersection() {
        let cyl_ray = CylindricalRay::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            1.0,
        );

        // Point on the ray
        let (inside, dist) = cyl_ray.point_intersection(Vector3::new(5.0, 0.0, 0.0));
        assert!(inside);
        assert!((dist - 5.0).abs() < 1e-6);

        // Point inside the cylinder
        let (inside, _) = cyl_ray.point_intersection(Vector3::new(5.0, 0.5, 0.0));
        assert!(inside);

        // Point outside the cylinder
        let (inside, _) = cyl_ray.point_intersection(Vector3::new(5.0, 2.0, 0.0));
        assert!(!inside);

        // Point behind the ray origin
        let (inside, _) = cyl_ray.point_intersection(Vector3::new(-1.0, 0.0, 0.0));
        assert!(!inside);
    }
}
