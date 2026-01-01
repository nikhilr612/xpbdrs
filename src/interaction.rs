//! Ray casting and mesh interaction system with optional spatial acceleration.

use raylib::prelude::*;
use std::collections::HashMap;

/// A ray cast from a point in a direction.
#[derive(Clone, Debug)]
pub struct Ray {
    /// Origin point of the ray.
    pub origin: Vector3,
    /// Normalized direction of the ray.
    pub direction: Vector3,
}

/// Simple spatial grid for accelerating ray queries.
/// Divides space into cells and stores vertex indices in each cell.
pub struct SpatialGrid {
    /// Cell size (determines grid resolution).
    cell_size: f32,
    /// Map from cell coordinates to vertex indices.
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    /// Create a new spatial grid with the given cell size.
    #[must_use]
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
        }
    }
    
    /// Build the spatial grid from a set of vertices.
    pub fn build(&mut self, vertices: &[crate::mesh::Vertex]) {
        self.cells.clear();
        for (idx, vertex) in vertices.iter().enumerate() {
            let cell = self.position_to_cell(vertex.position);
            self.cells.entry(cell).or_default().push(idx);
        }
    }
    
    /// Convert a position to cell coordinates.
    fn position_to_cell(&self, pos: Vector3) -> (i32, i32, i32) {
        (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        )
    }
    
    /// Get all vertex indices that might be within distance `radius` of the ray.
    /// Returns indices of vertices in cells that the ray passes through or near.
    pub fn query_ray(&self, ray: &CylindricalRay, max_distance: f32) -> Vec<usize> {
        let mut result = Vec::new();
        let mut visited: std::collections::HashSet<(i32, i32, i32)> = std::collections::HashSet::new();
        
        // Sample points along the ray and collect nearby cells
        let num_samples = ((max_distance / self.cell_size) * 2.0).ceil() as i32;
        let step = max_distance / num_samples.max(1) as f32;
        
        for i in 0..=num_samples {
            let t = i as f32 * step;
            let point = ray.ray.origin + ray.ray.direction * t;
            let center_cell = self.position_to_cell(point);
            
            // Check center cell and neighbors (to catch vertices on cell boundaries)
            let radius_cells = (ray.radius / self.cell_size).ceil() as i32;
            for dx in -radius_cells..=radius_cells {
                for dy in -radius_cells..=radius_cells {
                    for dz in -radius_cells..=radius_cells {
                        let cell = (center_cell.0 + dx, center_cell.1 + dy, center_cell.2 + dz);
                        if visited.insert(cell) {
                            if let Some(indices) = self.cells.get(&cell) {
                                result.extend(indices.iter().copied());
                            }
                        }
                    }
                }
            }
        }
        
        result
    }
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
    /// Computes the ray direction using standard perspective projection math.
    pub fn from_camera_mouse(
        camera: &Camera3D,
        screen_width: i32,
        screen_height: i32,
        mouse_x: i32,
        mouse_y: i32,
        radius: f32,
    ) -> Self {
        // Compute ray using standard perspective projection
        // 1. Convert mouse position to normalized device coordinates [-1, 1]
        let ndc_x = (2.0 * mouse_x as f32 / screen_width as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * mouse_y as f32 / screen_height as f32); // Y is inverted
        
        // 2. Get camera basis vectors
        let forward = (camera.target - camera.position).normalized();
        let right = forward.cross(camera.up).normalized();
        let up = right.cross(forward).normalized();
        
        // 3. Compute ray direction using FOV
        // Camera FOV is the vertical field of view in degrees
        let fov_rad = camera.fovy.to_radians();
        let aspect = screen_width as f32 / screen_height as f32;
        
        // Scale factors based on FOV
        let tan_half_fov = (fov_rad / 2.0).tan();
        let scale_y = tan_half_fov;
        let scale_x = tan_half_fov * aspect;
        
        // 4. Compute ray direction in world space
        let ray_direction = (forward + right * (ndc_x * scale_x) + up * (ndc_y * scale_y)).normalized();
        
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
    /// Force magnitude represents the base strength; actual force = magnitude / (distance_to_axis² + ε).
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
                const EPSILON: f32 = 1e-6;
                let falloff = 1.0 / (distance_to_ray_axis * distance_to_ray_axis + EPSILON);

                // Force along ray direction with inverse square falloff
                let force = ray.ray.direction * (force_magnitude * falloff);

                effect.affected_vertices.push(idx);
                effect.forces.push(force);
            }
        }

        effect
    }
    
    /// Compute interaction effect using spatial acceleration.
    ///
    /// More efficient for large meshes when a spatial grid has been precomputed.
    /// `max_ray_distance` limits how far along the ray to search for vertices.
    pub fn from_cylindrical_ray_accelerated(
        ray: &CylindricalRay,
        vertices: &[crate::mesh::Vertex],
        grid: &SpatialGrid,
        force_magnitude: f32,
        max_ray_distance: f32,
    ) -> Self {
        let mut effect = InteractionEffect::new();
        
        // Query spatial grid for candidate vertices
        let candidates = grid.query_ray(ray, max_ray_distance);
        
        for idx in candidates {
            if let Some(vertex) = vertices.get(idx) {
                let (is_inside, distance) = ray.point_intersection(vertex.position);

                if is_inside && distance >= 0.0 && distance <= max_ray_distance {
                    // Compute distance from ray axis
                    let to_point = vertex.position - ray.ray.origin;
                    let distance_to_ray_axis = (to_point - ray.ray.direction * to_point.dot(ray.ray.direction)).length();
                    
                    // Inverse square law falloff
                    const EPSILON: f32 = 1e-6;
                    let falloff = 1.0 / (distance_to_ray_axis * distance_to_ray_axis + EPSILON);

                    let force = ray.ray.direction * (force_magnitude * falloff);

                    effect.affected_vertices.push(idx);
                    effect.forces.push(force);
                }
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
    use crate::mesh::Vertex;

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
    
    fn create_test_vertices() -> Vec<Vertex> {
        vec![
            // Vertex on the ray axis at distance 5
            Vertex { position: Vector3::new(5.0, 0.0, 0.0), inv_mass: 1.0 },
            // Vertex inside cylinder at distance 5, offset 0.3 from axis
            Vertex { position: Vector3::new(5.0, 0.3, 0.0), inv_mass: 1.0 },
            // Vertex outside cylinder at distance 5, offset 2.0 from axis
            Vertex { position: Vector3::new(5.0, 2.0, 0.0), inv_mass: 1.0 },
            // Vertex behind ray origin
            Vertex { position: Vector3::new(-1.0, 0.0, 0.0), inv_mass: 1.0 },
        ]
    }
    
    #[test]
    fn test_from_cylindrical_ray_force_magnitude() {
        let cyl_ray = CylindricalRay::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            1.0,
        );
        
        let vertices = create_test_vertices();
        let effect = InteractionEffect::from_cylindrical_ray(&cyl_ray, &vertices, 100.0);
        
        // Should affect vertices 0 and 1 (inside cylinder)
        assert_eq!(effect.affected_vertices.len(), 2);
        assert!(effect.affected_vertices.contains(&0));
        assert!(effect.affected_vertices.contains(&1));
        
        // Forces should be along ray direction (positive x)
        for force in &effect.forces {
            assert!(force.x > 0.0, "Force should be along positive x direction");
            assert!(force.y.abs() < 1e-6, "Force y should be near zero");
            assert!(force.z.abs() < 1e-6, "Force z should be near zero");
        }
    }
    
    #[test]
    fn test_from_cylindrical_ray_falloff_behavior() {
        let cyl_ray = CylindricalRay::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            1.0,
        );
        
        let vertices = create_test_vertices();
        let effect = InteractionEffect::from_cylindrical_ray(&cyl_ray, &vertices, 100.0);
        
        // Find forces for vertex 0 (on axis) and vertex 1 (offset from axis)
        let idx_0 = effect.affected_vertices.iter().position(|&v| v == 0).unwrap();
        let idx_1 = effect.affected_vertices.iter().position(|&v| v == 1).unwrap();
        
        let force_on_axis = effect.forces[idx_0].length();
        let force_offset = effect.forces[idx_1].length();
        
        // Force on axis should be much larger (inverse square falloff)
        assert!(
            force_on_axis > force_offset * 10.0,
            "Force on axis ({}) should be much larger than offset force ({})",
            force_on_axis,
            force_offset
        );
    }
    
    #[test]
    fn test_from_cylindrical_ray_no_hit_cases() {
        let cyl_ray = CylindricalRay::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            0.1, // Small radius
        );
        
        // All vertices are offset more than 0.1 from the ray axis
        let vertices = vec![
            Vertex { position: Vector3::new(5.0, 0.5, 0.0), inv_mass: 1.0 },
            Vertex { position: Vector3::new(5.0, 0.0, 0.5), inv_mass: 1.0 },
            Vertex { position: Vector3::new(-1.0, 0.0, 0.0), inv_mass: 1.0 }, // Behind
        ];
        
        let effect = InteractionEffect::from_cylindrical_ray(&cyl_ray, &vertices, 100.0);
        
        assert!(
            effect.affected_vertices.is_empty(),
            "No vertices should be hit with small radius"
        );
    }
    
    #[test]
    fn test_spatial_grid_basic() {
        let vertices = create_test_vertices();
        let mut grid = SpatialGrid::new(1.0);
        grid.build(&vertices);
        
        let cyl_ray = CylindricalRay::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            1.0,
        );
        
        let candidates = grid.query_ray(&cyl_ray, 10.0);
        
        // Should find vertices near the ray
        assert!(!candidates.is_empty(), "Should find candidate vertices");
        assert!(candidates.contains(&0), "Should find vertex on ray axis");
        assert!(candidates.contains(&1), "Should find vertex near ray");
    }
    
    #[test]
    fn test_from_cylindrical_ray_accelerated() {
        let vertices = create_test_vertices();
        let mut grid = SpatialGrid::new(1.0);
        grid.build(&vertices);
        
        let cyl_ray = CylindricalRay::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            1.0,
        );
        
        let effect_normal = InteractionEffect::from_cylindrical_ray(&cyl_ray, &vertices, 100.0);
        let effect_accel = InteractionEffect::from_cylindrical_ray_accelerated(
            &cyl_ray, &vertices, &grid, 100.0, 10.0
        );
        
        // Both should produce the same affected vertices
        assert_eq!(
            effect_normal.affected_vertices.len(),
            effect_accel.affected_vertices.len(),
            "Accelerated version should find same number of vertices"
        );
    }
}
