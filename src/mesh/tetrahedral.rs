//! Tetrahedral mesh implementation.

use raylib::prelude::*;
use std::io::Write;
use tracing::{debug, info};

use super::common::{Result, TetrahedronId, Triangle, Vertex, dedup_with_warning};
use super::tgimport::TetgenParser;
use crate::constraint::Constraint;
use crate::mesh::{Edge, Tetrahedron};
use crate::xpbd::{ConstraintSet, XpbdState};

/// Values computed from tetrahedral constraints.
pub struct TetConstraintValues {
    /// Edge lengths for distance constraints.
    pub lengths: Vec<f32>,
    /// Tetrahedron volumes for volume constraints.
    pub volumes: Vec<f32>,
}

/// Struct to contain constraint data for teetrahedral meshes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TetConstraints {
    /// Edge constraints for distance preservation.
    pub edges: Vec<Edge>,
    /// Tetrahedral constraints for volume preservation.
    pub tetrahedra: Vec<Tetrahedron>,
}

impl ConstraintSet<Vec<Vertex>, TetConstraintValues> for TetConstraints {
    fn size(&self) -> usize {
        self.edges.len() + self.tetrahedra.len()
    }

    fn evaluate(&self, on: &Vec<Vertex>) -> TetConstraintValues {
        let lengths = self.edges.iter().map(|e| e.value(on)).collect();
        let volumes = self.tetrahedra.iter().map(|t| t.value(on)).collect();
        TetConstraintValues { lengths, volumes }
    }

    fn solve(
        &self,
        processor: crate::xpbd::ConstraintProcessor<Vec<Vertex>>,
        params: &crate::xpbd::XpbdParams,
        reference: &TetConstraintValues,
    ) {
        let dt2 = params.time_substep * params.time_substep;
        let _ = processor
            .process(
                self.edges.iter().zip(reference.lengths.iter().copied()),
                params.l_threshold_length * dt2,
                params.length_compliance / dt2,
            )
            .process(
                self.tetrahedra
                    .iter()
                    .zip(reference.volumes.iter().map(|v| v * params.p_volume)),
                params.l_threshold_volume * dt2,
                params.volume_compliance / dt2,
            );
    }
}

/// Struct to contain data of a delaunay tetrahedralized mesh.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Tetrahedral {
    /// Vertices of the tetrahedral mesh.
    pub vertices: Vec<Vertex>,
    /// Constraints for physics simulation.
    pub constraints: TetConstraints,
    /// Triangular faces of the mesh.
    pub faces: Vec<Triangle>,
}

impl Tetrahedral {
    /// Get the corner points of a tetrahedron by its ID.
    ///
    /// # Returns
    /// `None` if the tetrahedron ID is invalid or references invalid vertices.
    #[must_use]
    pub fn corners(&self, id: TetrahedronId) -> Option<[Vector3; 4]> {
        let tet = self.constraints.tetrahedra.get(id.0 as usize)?;
        Some([
            self.vertices[tet.indices[0]].position,
            self.vertices[tet.indices[1]].position,
            self.vertices[tet.indices[2]].position,
            self.vertices[tet.indices[3]].position,
        ])
    }

    /// Load tetrahedral mesh from tetgen files.
    ///
    /// # Errors
    /// Returns an error if files cannot be read or parsed.
    #[tracing::instrument]
    pub fn from_files(prefix: &str) -> Result<Self> {
        // Deduplication is now streamlined using custom Hash and Eq implementations
        let vertices = TetgenParser::load_vertices(prefix)?;
        let edges = dedup_with_warning(TetgenParser::load_edges(prefix)?, "edge");
        let face_triangles = TetgenParser::load_face_vertices(prefix)?;
        let (edges, faces) = TetgenParser::build_faces_with_edges(edges, face_triangles);
        let tetrahedra = dedup_with_warning(TetgenParser::load_tetrahedra(prefix)?, "tetrahedron");

        let result = Self {
            vertices,
            constraints: TetConstraints { edges, tetrahedra },
            faces,
        };

        info!(
            vertices = result.vertices.len(),
            edges = result.constraints.edges.len(),
            faces = result.faces.len(),
            tetrahedra = result.constraints.tetrahedra.len(),
            "Mesh loaded from tetgen files"
        );

        Ok(result)
    }

    /// Load tetrahedral mesh from bincode file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or deserialized.
    #[tracing::instrument]
    pub fn from_bincode(filename: &str) -> Result<Self> {
        let data = std::fs::read(filename)?;
        debug!("Deserializing {} bytes", data.len());
        let mesh: Self = bincode::deserialize(&data)?;
        Ok(mesh)
    }

    /// Export mesh to bincode format.
    ///
    /// # Errors
    /// Returns an error if serialization fails or file cannot be written.
    #[tracing::instrument(skip(self))]
    pub fn export_to_bincode(&self, output_path: &str) -> Result<()> {
        info!("Serializing to binary format");
        let encoded = bincode::serialize(self)?;

        let mut file = std::fs::File::create(output_path)?;
        file.write_all(&encoded)?;

        info!(
            output_path,
            size_bytes = encoded.len(),
            "Successfully exported mesh"
        );

        // Verify deserialization works
        debug!("Verifying serialized data");
        let _: Self = bincode::deserialize(&encoded)?;
        debug!("Verification successful");

        Ok(())
    }

    /// Draw wireframe of the mesh.
    pub fn draw_wireframe(&self, d3: &mut RaylibMode3D<RaylibDrawHandle>, color: Color) {
        // Draw explicit edges if available
        for edge in &self.constraints.edges {
            if let (Some(v1), Some(v2)) = (
                self.vertices.get((edge.0.0 - 1) as usize),
                self.vertices.get((edge.1.0 - 1) as usize),
            ) {
                let start = v1.position;
                let end = v2.position;
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
                    verts[0].position,
                    verts[1].position,
                    verts[2].position,
                    color,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn create_test_files(prefix: &str) {
        fs::write(
            format!("{prefix}.node"),
            "2 3 0 0\n1 0.0 0.0 0.0\n2 1.0 1.0 1.0\n",
        )
        .unwrap();
        fs::write(format!("{prefix}.edge"), "1 0\n1 1 2\n").unwrap();
        fs::write(format!("{prefix}.face"), "1 0\n1 1 2 1\n").unwrap();
        fs::write(format!("{prefix}.ele"), "1 4 0\n1 1 2 1 2\n").unwrap();
    }

    #[test]
    fn test_parse() {
        let prefix = "test";
        create_test_files(prefix);

        let mesh = Tetrahedral::from_files(prefix).unwrap();
        assert_eq!(mesh.vertices.len(), 2);
        assert_eq!(mesh.constraints.edges.len(), 1);
        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.constraints.tetrahedra.len(), 1);

        // Cleanup
        for ext in &["node", "edge", "face", "ele"] {
            let _ = fs::remove_file(format!("{prefix}.{ext}"));
        }
    }
}
