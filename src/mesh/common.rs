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

/// Common trait for mesh types providing core mesh operations.
pub trait Mesh {
    /// Translate the entire mesh by a vector.
    fn translate(&mut self, by: Vector3);

    /// Get bounding box of the mesh as (min, max) corners.
    fn bounding_box(&self) -> (Vector3, Vector3);
}
