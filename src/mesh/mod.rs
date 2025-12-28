//! Mesh module providing various mesh types and common functionality.
//!
//! This module contains different mesh implementations and shared utilities:
//! - Common data structures (vertices, edges, triangles, etc.)
//! - Tetrahedral mesh implementation
//! - Shared traits for translation and bounding box operations

pub mod common;
pub mod tetrahedral;
pub mod tgimport;
pub mod triangular;

// Re-export common types for convenience
pub use common::{
    Edge, EdgeId, Result, Spatial, Tetrahedron, TetrahedronId, Triangle, Vertex, VertexId,
    dedup_with_warning,
};

// Re-export specific mesh types
pub use tetrahedral::Tetrahedral;
pub use triangular::TriangulatedSurface;
