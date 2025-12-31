//! Module for parsing tetgen file formats.
//!
//! This module handles the parsing of tetgen output files (.node, .ele, .edge, .face)
//! and provides functionality to load tetrahedral mesh data from these files.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tracing::debug;

use super::common::{Edge, EdgeId, Result, Tetrahedron, Triangle, Vertex, VertexId};

/// Tetgen file parser for loading mesh data from tetgen output files.
pub struct TetgenParser;

impl TetgenParser {
    /// Parse a generic tetgen file format.
    ///
    /// The first line contains the count of items, followed by data lines.
    /// Comments (lines starting with #) and empty lines are ignored.
    fn parse_file<T>(filename: &str, processor: impl Fn(&[&str]) -> Result<T>) -> Result<Vec<T>> {
        let file = File::open(filename)?;

        let mut lines = BufReader::new(file).lines();
        let count: usize = lines
            .next()
            .ok_or("Empty file")??
            .split_whitespace()
            .next()
            .ok_or("Invalid first line")?
            .parse()?;

        debug!("Expecting {} items from file", count);

        lines
            .map_while(std::result::Result::ok)
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                let tokens: Vec<&str> = line.split_whitespace().collect();
                processor(&tokens)
            })
            .take(count)
            .collect()
    }

    /// Parse a sequence of indices from tokens starting at a given position.
    fn parse_indices(tokens: &[&str], start: usize, count: usize) -> Result<Vec<u32>> {
        tokens[start..start + count]
            .iter()
            .map(|&t| t.parse().map_err(Into::into))
            .collect()
    }

    /// Load vertices from a .node file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or parsed.
    #[tracing::instrument]
    pub fn load_vertices(prefix: &str) -> Result<Vec<Vertex>> {
        Self::parse_file(&format!("{prefix}.node"), |tokens| {
            let coords: Vec<f32> = tokens[1..4]
                .iter()
                .map(|&t| t.parse().map_err(Into::into))
                .collect::<Result<_>>()?;
            Ok(Vertex {
                position: glam::Vec3::new(coords[0], coords[1], coords[2]),
                inv_mass: 1.0,
            })
        })
    }

    /// Load edges from a .edge file if it exists.
    ///
    /// # Errors
    /// Returns an error if the file exists but cannot be parsed.
    #[tracing::instrument]
    pub fn load_edges(prefix: &str) -> Result<Vec<Edge>> {
        let edge_file = format!("{prefix}.edge");
        if Path::new(&edge_file).exists() {
            debug!("Loading edges from {}", edge_file);
            Self::parse_file(&edge_file, |tokens| {
                let ids = Self::parse_indices(tokens, 1, 2)?;
                Ok(Edge(VertexId(ids[0]), VertexId(ids[1])))
            })
        } else {
            debug!("No edge file found, using empty edge list");
            Ok(Vec::new())
        }
    }

    /// Load face vertices from a .face file if it exists.
    ///
    /// # Errors
    /// Returns an error if the file exists but cannot be parsed.
    #[tracing::instrument]
    pub fn load_face_vertices(prefix: &str) -> Result<Vec<[VertexId; 3]>> {
        let face_file = format!("{prefix}.face");
        if Path::new(&face_file).exists() {
            debug!("Loading faces from {}", face_file);
            Self::parse_file(&face_file, |tokens| {
                let ids = Self::parse_indices(tokens, 1, 3)?;
                Ok([VertexId(ids[0]), VertexId(ids[1]), VertexId(ids[2])])
            })
        } else {
            debug!("No face file found, using empty face list");
            Ok(Vec::new())
        }
    }

    /// Load tetrahedra from a .ele file if it exists.
    ///
    /// # Errors
    /// Returns an error if the file exists but cannot be parsed.
    #[tracing::instrument]
    pub fn load_tetrahedra(prefix: &str) -> Result<Vec<Tetrahedron>> {
        let ele_file = format!("{prefix}.ele");
        if Path::new(&ele_file).exists() {
            debug!("Loading tetrahedra from {}", ele_file);
            Self::parse_file(&ele_file, |tokens| {
                let ids = Self::parse_indices(tokens, 1, 4)?;
                Ok(Tetrahedron {
                    indices: [
                        VertexId(ids[0]),
                        VertexId(ids[1]),
                        VertexId(ids[2]),
                        VertexId(ids[3]),
                    ],
                })
            })
        } else {
            debug!("No element file found, using empty tetrahedra list");
            Ok(Vec::new())
        }
    }

    /// Build triangular faces with associated edge IDs.
    ///
    /// This function creates a mapping from edges to their IDs and then
    /// constructs triangular faces with references to the appropriate edges.
    ///
    /// # Panics
    /// Panics if the number of edges exceeds `u32::MAX`.
    #[tracing::instrument(skip(edges, face_triangles), fields(edges_count = edges.len(), faces_count = face_triangles.len()))]
    pub fn build_faces_with_edges(
        edges: Vec<Edge>,
        face_triangles: Vec<[VertexId; 3]>,
    ) -> (Vec<Edge>, Vec<Triangle>) {
        let edge_map: HashMap<Edge, EdgeId> = edges
            .iter()
            .enumerate()
            .map(|(i, edge)| {
                (
                    edge.clone(),
                    EdgeId(
                        i.try_into()
                            .expect("Edge count should not be more than u32::MAX+1"),
                    ),
                )
            })
            .collect();

        let faces = face_triangles
            .into_iter()
            .map(|verts| {
                let edge_ids = [
                    Self::find_existing_edge_id(&edge_map, verts[0], verts[1]),
                    Self::find_existing_edge_id(&edge_map, verts[1], verts[2]),
                    Self::find_existing_edge_id(&edge_map, verts[2], verts[0]),
                ];

                Triangle {
                    verts,
                    edges: edge_ids,
                }
            })
            .collect();

        (edges, faces)
    }

    /// Find an existing edge ID in the edge mapping.
    ///
    /// Uses the custom Hash and Eq implementation for Edge to handle
    /// undirected edge lookups automatically.
    fn find_existing_edge_id(
        edge_map: &HashMap<Edge, EdgeId>,
        v1: VertexId,
        v2: VertexId,
    ) -> Option<EdgeId> {
        let edge = Edge(v1, v2);
        // With custom Hash and Eq, no need to manually normalize
        edge_map.get(&edge).copied()
    }
}
