//! General constraints for physics simulation.

use std::ops::Index;

use crate::mesh::{Edge, Tetrahedron, Vertex, VertexId};
use raylib::math::Vector3;

/// The value and gradient of an n-ary constraint.
#[derive(Debug)]
pub struct ValueGrad<const ARITY: usize> {
    /// Contraint value.
    pub value: f32,
    /// Constraint gradient with respect to each vertex.
    pub grad: [Vector3; ARITY],
    /// Indices of participating vertices.
    pub participants: [VertexId; ARITY],
}

/// Trait for a general constraint with `ARITY` participants.
pub trait Constraint<const ARITY: usize> {
    /// Evaluate this constraint and obtain the corresponding value.
    fn value<V>(&self, vertices: &V) -> f32
    where
        V: Index<VertexId, Output = Vertex>;

    /// Evaluate this constraint and obtain both the value and gradient.
    fn value_and_grad<V>(&self, vertices: &V) -> ValueGrad<ARITY>
    where
        V: Index<VertexId, Output = Vertex>;
}

/// Binary edge constraint.
impl Constraint<2> for Edge {
    // Value for edge constraint is the distance between participating vertices.
    fn value<V>(&self, vertices: &V) -> f32
    where
        V: Index<VertexId>,
        <V as Index<VertexId>>::Output: Into<Vertex> + Copy,
    {
        let v1 = vertices[self.0].into();
        let v2 = vertices[self.1].into();
        let delta = v1.position - v2.position;
        delta.length()
    }

    fn value_and_grad<V>(&self, vertices: &V) -> ValueGrad<2>
    where
        V: Index<VertexId>,
        <V as Index<VertexId>>::Output: Into<Vertex> + Copy,
    {
        let v1: Vertex = vertices[self.0].into();
        let v2: Vertex = vertices[self.1].into();
        let delta = v1.position - v2.position;
        let distance = delta.length();

        // Gradient of distance with respect to each vertex
        // d(||p1 - p2||)/dp1 = (p1 - p2) / ||p1 - p2||
        // d(||p1 - p2||)/dp2 = -(p1 - p2) / ||p1 - p2||
        let grad = if distance > 1e-8 {
            delta / distance
        } else {
            // Handle degenerate case where vertices are at the same position
            Vector3::new(1.0, 0.0, 0.0)
        };

        ValueGrad {
            value: distance,
            grad: [grad, -grad],
            participants: [self.0, self.1],
        }
    }
}

impl Constraint<4> for Tetrahedron {
    fn value<V>(&self, vertices: &V) -> f32
    where
        V: Index<VertexId>,
        <V as Index<VertexId>>::Output: Into<Vertex> + Copy,
    {
        let v0 = vertices[self.indices[0]].into();
        let v1 = vertices[self.indices[1]].into();
        let v2 = vertices[self.indices[2]].into();
        let v3 = vertices[self.indices[3]].into();

        let a = v1.position - v0.position;
        let b = v2.position - v0.position;
        let c = v3.position - v0.position;

        (a.cross(b)).dot(c) / 6.0
    }

    fn value_and_grad<V>(&self, vertices: &V) -> ValueGrad<4>
    where
        V: Index<VertexId>,
        <V as Index<VertexId>>::Output: Into<Vertex> + Copy,
    {
        let v0 = vertices[self.indices[0]].into().position;
        let v1 = vertices[self.indices[1]].into().position;
        let v2 = vertices[self.indices[2]].into().position;
        let v3 = vertices[self.indices[3]].into().position;

        let cross_ab = (v1 - v0).cross(v2 - v0);
        let volume = cross_ab.dot(v3 - v0) / 6.0;

        let grad_v0 = (v3 - v1).cross(v2 - v1) / 6.0;
        let grad_v1 = (v2 - v0).cross(v3 - v0) / 6.0;
        let grad_v2 = (v3 - v0).cross(v1 - v0) / 6.0;
        let grad_v3 = cross_ab / 6.0;

        ValueGrad {
            value: volume,
            grad: [grad_v0, grad_v1, grad_v2, grad_v3],
            participants: self.indices,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raylib::math::Vector3;
    use std::collections::HashMap;

    /// Create a mock vertex storage for testing
    struct MockVertices {
        vertices: HashMap<VertexId, Vertex>,
    }

    impl MockVertices {
        fn new() -> Self {
            Self {
                vertices: HashMap::new(),
            }
        }

        fn insert(&mut self, id: VertexId, position: Vector3) {
            self.vertices.insert(id, Vertex { position });
        }
    }

    impl Index<VertexId> for MockVertices {
        type Output = Vertex;

        fn index(&self, index: VertexId) -> &Self::Output {
            &self.vertices[&index]
        }
    }

    const EPSILON: f32 = 1e-6;

    fn assert_vector3_eq(a: Vector3, b: Vector3, epsilon: f32) {
        assert!(
            (a.x - b.x).abs() < epsilon
                && (a.y - b.y).abs() < epsilon
                && (a.z - b.z).abs() < epsilon,
            "Expected {:?}, got {:?}, difference: {:?}",
            b,
            a,
            a - b
        );
    }

    #[test]
    fn test_edge_constraint_value_basic() {
        let mut vertices = MockVertices::new();
        vertices.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices.insert(VertexId(1), Vector3::new(3.0, 4.0, 0.0));

        let edge = Edge(VertexId(0), VertexId(1));
        let value = edge.value(&vertices);

        // Distance should be sqrt(3^2 + 4^2) = 5.0
        assert!((value - 5.0).abs() < EPSILON, "Expected 5.0, got {value}");
    }

    #[test]
    fn test_edge_constraint_value_zero_distance() {
        let mut vertices = MockVertices::new();
        vertices.insert(VertexId(0), Vector3::new(1.0, 2.0, 3.0));
        vertices.insert(VertexId(1), Vector3::new(1.0, 2.0, 3.0));

        let edge = Edge(VertexId(0), VertexId(1));
        let value = edge.value(&vertices);

        assert!(value.abs() < EPSILON, "Expected 0.0, got {value}");
    }

    #[test]
    fn test_edge_constraint_value_unit_vectors() {
        let mut vertices = MockVertices::new();
        vertices.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices.insert(VertexId(1), Vector3::new(1.0, 0.0, 0.0));

        let edge = Edge(VertexId(0), VertexId(1));
        let value = edge.value(&vertices);

        assert!((value - 1.0).abs() < EPSILON, "Expected 1.0, got {value}");
    }

    #[test]
    fn test_edge_constraint_value_and_grad_basic() {
        let mut vertices = MockVertices::new();
        vertices.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices.insert(VertexId(1), Vector3::new(3.0, 4.0, 0.0));

        let edge = Edge(VertexId(0), VertexId(1));
        let result = edge.value_and_grad(&vertices);

        // Value should be 5.0
        assert!(
            (result.value - 5.0).abs() < EPSILON,
            "Expected value 5.0, got {}",
            result.value
        );

        // Gradient should be normalized direction vector
        // delta = v1 - v2 = (0,0,0) - (3,4,0) = (-3,-4,0)
        // normalized delta = (-0.6, -0.8, 0.0)
        let expected_grad = Vector3::new(-0.6, -0.8, 0.0);
        assert_vector3_eq(result.grad[0], expected_grad, EPSILON);
        assert_vector3_eq(result.grad[1], -expected_grad, EPSILON);

        // Participants should match
        assert_eq!(result.participants, [VertexId(0), VertexId(1)]);
    }

    #[test]
    fn test_edge_constraint_grad_degenerate_case() {
        let mut vertices = MockVertices::new();
        vertices.insert(VertexId(0), Vector3::new(1.0, 2.0, 3.0));
        vertices.insert(VertexId(1), Vector3::new(1.0, 2.0, 3.0));

        let edge = Edge(VertexId(0), VertexId(1));
        let result = edge.value_and_grad(&vertices);

        // Value should be 0
        assert!(
            result.value.abs() < EPSILON,
            "Expected value 0.0, got {}",
            result.value
        );

        // Gradient should be the fallback vector
        assert_vector3_eq(result.grad[0], Vector3::new(1.0, 0.0, 0.0), EPSILON);
        assert_vector3_eq(result.grad[1], Vector3::new(-1.0, 0.0, 0.0), EPSILON);
    }

    #[test]
    fn test_edge_constraint_grad_symmetry() {
        let mut vertices = MockVertices::new();
        vertices.insert(VertexId(0), Vector3::new(1.0, 2.0, 3.0));
        vertices.insert(VertexId(1), Vector3::new(4.0, 6.0, 8.0));

        let edge = Edge(VertexId(0), VertexId(1));
        let result = edge.value_and_grad(&vertices);

        // Gradients should be opposite
        assert_vector3_eq(result.grad[0], -result.grad[1], EPSILON);
    }

    #[test]
    fn test_edge_constraint_grad_magnitude() {
        let mut vertices = MockVertices::new();
        vertices.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices.insert(VertexId(1), Vector3::new(5.0, 12.0, 0.0));

        let edge = Edge(VertexId(0), VertexId(1));
        let result = edge.value_and_grad(&vertices);

        // Gradient magnitude should be 1 (unit vector)
        let grad_magnitude = result.grad[0].length();
        assert!(
            (grad_magnitude - 1.0).abs() < EPSILON,
            "Expected gradient magnitude 1.0, got {grad_magnitude}"
        );
    }

    #[test]
    fn test_tetrahedron_constraint_value_unit_cube() {
        let mut vertices = MockVertices::new();
        vertices.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices.insert(VertexId(1), Vector3::new(1.0, 0.0, 0.0));
        vertices.insert(VertexId(2), Vector3::new(0.0, 1.0, 0.0));
        vertices.insert(VertexId(3), Vector3::new(0.0, 0.0, 1.0));

        let tetrahedron = Tetrahedron {
            indices: [VertexId(0), VertexId(1), VertexId(2), VertexId(3)],
        };
        let value = tetrahedron.value(&vertices);

        // Volume should be 1/6
        let expected_volume = 1.0 / 6.0;
        assert!(
            (value - expected_volume).abs() < EPSILON,
            "Expected {expected_volume}, got {value}"
        );
    }

    #[test]
    fn test_tetrahedron_constraint_value_degenerate() {
        let mut vertices = MockVertices::new();
        // All points in the same plane (z=0)
        vertices.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices.insert(VertexId(1), Vector3::new(1.0, 0.0, 0.0));
        vertices.insert(VertexId(2), Vector3::new(0.0, 1.0, 0.0));
        vertices.insert(VertexId(3), Vector3::new(1.0, 1.0, 0.0));

        let tetrahedron = Tetrahedron {
            indices: [VertexId(0), VertexId(1), VertexId(2), VertexId(3)],
        };
        let value = tetrahedron.value(&vertices);

        // Volume should be 0 (degenerate tetrahedron)
        assert!(value.abs() < EPSILON, "Expected 0.0, got {value}");
    }

    #[test]
    fn test_tetrahedron_constraint_value_regular() {
        let mut vertices = MockVertices::new();
        let h = (2.0_f32 / 3.0).sqrt(); // Height of regular tetrahedron with edge length sqrt(2)
        vertices.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices.insert(VertexId(1), Vector3::new(1.0, 0.0, 0.0));
        vertices.insert(VertexId(2), Vector3::new(0.5, (3.0_f32).sqrt() / 2.0, 0.0));
        vertices.insert(VertexId(3), Vector3::new(0.5, (3.0_f32).sqrt() / 6.0, h));

        let tetrahedron = Tetrahedron {
            indices: [VertexId(0), VertexId(1), VertexId(2), VertexId(3)],
        };
        let value = tetrahedron.value(&vertices);

        // Should be positive volume (correct orientation)
        assert!(value > 0.0, "Expected positive volume, got {value}");
    }

    #[test]
    fn test_tetrahedron_constraint_value_and_grad_basic() {
        let mut vertices = MockVertices::new();
        vertices.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices.insert(VertexId(1), Vector3::new(1.0, 0.0, 0.0));
        vertices.insert(VertexId(2), Vector3::new(0.0, 1.0, 0.0));
        vertices.insert(VertexId(3), Vector3::new(0.0, 0.0, 1.0));

        let tetrahedron = Tetrahedron {
            indices: [VertexId(0), VertexId(1), VertexId(2), VertexId(3)],
        };
        let result = tetrahedron.value_and_grad(&vertices);

        // Value should be 1/6
        let expected_volume = 1.0 / 6.0;
        assert!(
            (result.value - expected_volume).abs() < EPSILON,
            "Expected {}, got {}",
            expected_volume,
            result.value
        );

        // Participants should match
        assert_eq!(
            result.participants,
            [VertexId(0), VertexId(1), VertexId(2), VertexId(3)]
        );

        // Gradient sum should be zero (translation invariance)
        let grad_sum = result.grad[0] + result.grad[1] + result.grad[2] + result.grad[3];
        assert_vector3_eq(grad_sum, Vector3::zero(), EPSILON);
    }

    #[test]
    fn test_tetrahedron_constraint_grad_translation_invariance() {
        let mut vertices1 = MockVertices::new();
        vertices1.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices1.insert(VertexId(1), Vector3::new(1.0, 0.0, 0.0));
        vertices1.insert(VertexId(2), Vector3::new(0.0, 1.0, 0.0));
        vertices1.insert(VertexId(3), Vector3::new(0.0, 0.0, 1.0));

        let mut vertices2 = MockVertices::new();
        let offset = Vector3::new(5.0, 3.0, -2.0);
        vertices2.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0) + offset);
        vertices2.insert(VertexId(1), Vector3::new(1.0, 0.0, 0.0) + offset);
        vertices2.insert(VertexId(2), Vector3::new(0.0, 1.0, 0.0) + offset);
        vertices2.insert(VertexId(3), Vector3::new(0.0, 0.0, 1.0) + offset);

        let tetrahedron = Tetrahedron {
            indices: [VertexId(0), VertexId(1), VertexId(2), VertexId(3)],
        };

        let result1 = tetrahedron.value_and_grad(&vertices1);
        let result2 = tetrahedron.value_and_grad(&vertices2);

        // Values should be the same
        assert!(
            (result1.value - result2.value).abs() < EPSILON,
            "Values should be translation invariant: {} vs {}",
            result1.value,
            result2.value
        );

        // Gradients should be the same
        for i in 0..4 {
            assert_vector3_eq(result1.grad[i], result2.grad[i], EPSILON);
        }
    }

    #[test]
    fn test_tetrahedron_constraint_grad_rotation_covariance() {
        let mut vertices1 = MockVertices::new();
        vertices1.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices1.insert(VertexId(1), Vector3::new(1.0, 0.0, 0.0));
        vertices1.insert(VertexId(2), Vector3::new(0.0, 1.0, 0.0));
        vertices1.insert(VertexId(3), Vector3::new(0.0, 0.0, 1.0));

        // 90 degree rotation around z-axis
        let mut vertices2 = MockVertices::new();
        vertices2.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices2.insert(VertexId(1), Vector3::new(0.0, 1.0, 0.0));
        vertices2.insert(VertexId(2), Vector3::new(-1.0, 0.0, 0.0));
        vertices2.insert(VertexId(3), Vector3::new(0.0, 0.0, 1.0));

        let tetrahedron = Tetrahedron {
            indices: [VertexId(0), VertexId(1), VertexId(2), VertexId(3)],
        };

        let result1 = tetrahedron.value_and_grad(&vertices1);
        let result2 = tetrahedron.value_and_grad(&vertices2);

        // Values should be the same (rotation invariant)
        assert!(
            (result1.value - result2.value).abs() < EPSILON,
            "Values should be rotation invariant: {} vs {}",
            result1.value,
            result2.value
        );
    }

    #[test]
    fn test_tetrahedron_constraint_grad_finite_differences() {
        let mut vertices = MockVertices::new();
        vertices.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices.insert(VertexId(1), Vector3::new(1.0, 0.0, 0.0));
        vertices.insert(VertexId(2), Vector3::new(0.0, 1.0, 0.0));
        vertices.insert(VertexId(3), Vector3::new(0.0, 0.0, 1.0));

        let tetrahedron = Tetrahedron {
            indices: [VertexId(0), VertexId(1), VertexId(2), VertexId(3)],
        };

        let result = tetrahedron.value_and_grad(&vertices);
        let h = 1e-5;

        // Test gradient for vertex 1 in x direction
        let mut vertices_plus = vertices.vertices.clone();
        let mut vertices_minus = vertices.vertices.clone();

        vertices_plus.get_mut(&VertexId(1)).unwrap().position.x += h;
        vertices_minus.get_mut(&VertexId(1)).unwrap().position.x -= h;

        let vertices_plus = MockVertices {
            vertices: vertices_plus,
        };
        let vertices_minus = MockVertices {
            vertices: vertices_minus,
        };

        let value_plus = tetrahedron.value(&vertices_plus);
        let value_minus = tetrahedron.value(&vertices_minus);

        let finite_diff_grad = (value_plus - value_minus) / (2.0 * h);
        let analytical_grad = result.grad[1].x;

        assert!(
            (finite_diff_grad - analytical_grad).abs() < 1e-2,
            "Finite difference gradient {finite_diff_grad} doesn't match analytical gradient {analytical_grad}"
        );
    }

    #[test]
    fn test_edge_constraint_grad_finite_differences() {
        let mut vertices = MockVertices::new();
        vertices.insert(VertexId(0), Vector3::new(1.0, 2.0, 3.0));
        vertices.insert(VertexId(1), Vector3::new(4.0, 5.0, 6.0));

        let edge = Edge(VertexId(0), VertexId(1));
        let result = edge.value_and_grad(&vertices);
        let h = 1e-5;

        // Test gradient for vertex 0 in y direction
        let mut vertices_plus = vertices.vertices.clone();
        let mut vertices_minus = vertices.vertices.clone();

        vertices_plus.get_mut(&VertexId(0)).unwrap().position.y += h;
        vertices_minus.get_mut(&VertexId(0)).unwrap().position.y -= h;

        let vertices_plus = MockVertices {
            vertices: vertices_plus,
        };
        let vertices_minus = MockVertices {
            vertices: vertices_minus,
        };

        let value_plus = edge.value(&vertices_plus);
        let value_minus = edge.value(&vertices_minus);

        let finite_diff_grad = (value_plus - value_minus) / (2.0 * h);
        let analytical_grad = result.grad[0].y;

        assert!(
            (finite_diff_grad - analytical_grad).abs() < 5e-2,
            "Finite difference gradient {finite_diff_grad} doesn't match analytical gradient {analytical_grad}"
        );
    }

    #[test]
    fn test_tetrahedron_constraint_negative_volume() {
        let mut vertices = MockVertices::new();
        // Inverted orientation to get negative volume
        vertices.insert(VertexId(0), Vector3::new(0.0, 0.0, 0.0));
        vertices.insert(VertexId(1), Vector3::new(0.0, 1.0, 0.0));
        vertices.insert(VertexId(2), Vector3::new(1.0, 0.0, 0.0));
        vertices.insert(VertexId(3), Vector3::new(0.0, 0.0, 1.0));

        let tetrahedron = Tetrahedron {
            indices: [VertexId(0), VertexId(1), VertexId(2), VertexId(3)],
        };

        let result = tetrahedron.value_and_grad(&vertices);

        // Value should be negative (signed volume for inverted orientation)
        assert!(
            result.value < 0.0,
            "Expected negative volume, got {}",
            result.value
        );

        // Gradient sum should still be zero
        let grad_sum = result.grad[0] + result.grad[1] + result.grad[2] + result.grad[3];
        assert_vector3_eq(grad_sum, Vector3::zero(), EPSILON);
    }
}
