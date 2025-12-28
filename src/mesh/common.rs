//! Common data structures and utilities shared across different mesh types.

use raylib::prelude::*;
use std::collections::HashSet;
use tracing::{debug, warn};

/// Default inverse mass value for vertices.
fn default_inv_mass() -> f32 {
    1.0
}

/// A vertex in 3D space with position and inverse mass.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Vertex {
    /// 3D position of the vertex.
    pub position: Vector3,
    /// Inverse mass (1/mass) of the vertex.
    #[serde(default = "default_inv_mass")]
    pub inv_mass: f32,
}

/// Unique identifier for a vertex.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VertexId(pub u32);

/// An edge connecting two vertices.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Edge(pub VertexId, pub VertexId);

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        let (a1, a2) = if self.0.0 <= self.1.0 {
            (self.0, self.1)
        } else {
            (self.1, self.0)
        };
        let (b1, b2) = if other.0.0 <= other.1.0 {
            (other.0, other.1)
        } else {
            (other.1, other.0)
        };
        a1 == b1 && a2 == b2
    }
}

impl Eq for Edge {}

impl std::hash::Hash for Edge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let (v1, v2) = if self.0.0 <= self.1.0 {
            (self.0, self.1)
        } else {
            (self.1, self.0)
        };
        v1.hash(state);
        v2.hash(state);
    }
}

/// Unique identifier for an edge.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EdgeId(pub u32);

/// A triangular face defined by three vertex indices and their edges.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Triangle {
    /// Three vertex indices forming the triangle.
    pub verts: [VertexId; 3],
    /// Three edge indices connecting the vertices (None if no constraint exists).
    /// This is particularly useful for determining if a triangle is "torn" by checking the status of each of its edge constraints.
    pub edges: [Option<EdgeId>; 3],
}

/// A tetrahedron defined by four vertex indices.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Tetrahedron {
    /// Four vertex indices forming the tetrahedron.
    pub indices: [VertexId; 4],
}

impl PartialEq for Tetrahedron {
    fn eq(&self, other: &Self) -> bool {
        let mut a = self.indices;
        let mut b = other.indices;
        a.sort_by_key(|id| id.0);
        b.sort_by_key(|id| id.0);
        a == b
    }
}

impl Eq for Tetrahedron {}

impl std::hash::Hash for Tetrahedron {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut sorted_indices = self.indices;
        sorted_indices.sort_by_key(|id| id.0);
        sorted_indices.hash(state);
    }
}

/// Unique identifier for a tetrahedron.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TetrahedronId(pub u32);

/// Result type for mesh operations.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Generic function to deduplicate a collection using `HashSet`.
#[tracing::instrument(skip(items), fields(original_count = items.len()))]
pub fn dedup_with_warning<T>(items: Vec<T>, item_name: &str) -> Vec<T>
where
    T: std::hash::Hash + Eq + Clone,
{
    let original_count = items.len();

    let deduped: Vec<T> = items
        .into_iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let duplicate_count = original_count - deduped.len();
    if duplicate_count > 0 {
        warn!(
            "Found {} duplicate {} constraints",
            duplicate_count, item_name
        );
    } else {
        debug!("No duplicate {} constraints found", item_name);
    }

    deduped
}

/// Index implementation for accessing vertices by ID.
impl std::ops::Index<VertexId> for Vec<Vertex> {
    type Output = Vertex;

    fn index(&self, index: VertexId) -> &Self::Output {
        self.get((index.0 - 1) as usize).unwrap_or_else(|| {
            panic!(
                "Invalid vertex id: {}, only {} available.",
                index.0,
                self.len()
            )
        })
    }
}

/// Mutable index implementation for accessing vertices by ID.
impl std::ops::IndexMut<VertexId> for Vec<Vertex> {
    fn index_mut(&mut self, index: VertexId) -> &mut Self::Output {
        let len = self.len();
        self.get_mut((index.0 - 1) as usize)
            .unwrap_or_else(|| panic!("Invalid vertex id: {}, only {} available.", index.0, len))
    }
}

/// A convenience trait to aggregate "spatial" operations on collections of vertices.
/// This trait is not intended to be replete with all possible spatial operations, but is instead a conservative interface.
pub trait Spatial {
    /// Translate all vertices by a vector.
    fn translate(&mut self, by: Vector3);

    /// Get bounding box of the vertices as (min, max) corners.
    fn bounding_box(&self) -> (Vector3, Vector3);
}

impl Spatial for Vec<Vertex> {
    fn translate(&mut self, by: Vector3) {
        for vertex in self {
            vertex.position += by;
        }
    }

    fn bounding_box(&self) -> (Vector3, Vector3) {
        if self.is_empty() {
            return (Vector3::zero(), Vector3::zero());
        }

        let mut min = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for vertex in self {
            min.x = min.x.min(vertex.position.x);
            min.y = min.y.min(vertex.position.y);
            min.z = min.z.min(vertex.position.z);
            max.x = max.x.max(vertex.position.x);
            max.y = max.y.max(vertex.position.y);
            max.z = max.z.max(vertex.position.z);
        }

        (min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_translate() {
        let mut vertices = vec![
            Vertex {
                position: Vector3::new(0.0, 0.0, 0.0),
                inv_mass: 1.0,
            },
            Vertex {
                position: Vector3::new(1.0, 2.0, 3.0),
                inv_mass: 0.5,
            },
        ];

        let translation = Vector3::new(10.0, 20.0, 30.0);
        let original_positions: Vec<_> = vertices.iter().map(|v| v.position).collect();

        vertices.translate(translation);

        for (i, vertex) in vertices.iter().enumerate() {
            let expected = original_positions[i] + translation;
            assert!((vertex.position.x - expected.x).abs() < f32::EPSILON);
            assert!((vertex.position.y - expected.y).abs() < f32::EPSILON);
            assert!((vertex.position.z - expected.z).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_spatial_bounding_box() {
        let vertices = vec![
            Vertex {
                position: Vector3::new(-1.0, -2.0, -3.0),
                inv_mass: 1.0,
            },
            Vertex {
                position: Vector3::new(4.0, 5.0, 6.0),
                inv_mass: 1.0,
            },
            Vertex {
                position: Vector3::new(2.0, 1.0, 0.0),
                inv_mass: 1.0,
            },
        ];

        let (min, max) = vertices.bounding_box();

        assert_eq!(min, Vector3::new(-1.0, -2.0, -3.0));
        assert_eq!(max, Vector3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_spatial_bounding_box_single_vertex() {
        let vertices = vec![Vertex {
            position: Vector3::new(42.0, -17.0, 99.0),
            inv_mass: 1.0,
        }];

        let (min, max) = vertices.bounding_box();

        assert_eq!(min, Vector3::new(42.0, -17.0, 99.0));
        assert_eq!(max, Vector3::new(42.0, -17.0, 99.0));
    }

    #[test]
    fn test_spatial_bounding_box_empty() {
        let vertices: Vec<Vertex> = vec![];

        let (min, max) = vertices.bounding_box();

        assert_eq!(min, Vector3::zero());
        assert_eq!(max, Vector3::zero());
    }

    #[test]
    fn test_spatial_translate_empty() {
        let mut vertices: Vec<Vertex> = vec![];
        vertices.translate(Vector3::new(1.0, 2.0, 3.0));
        // Should not panic and remain empty
        assert!(vertices.is_empty());
    }
}
