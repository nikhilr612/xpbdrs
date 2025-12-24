//! Module to handle tetrahedral meshes.

use raylib::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::{Index, IndexMut};
use std::path::Path;

fn default_inv_mass() -> f32 {
    1.0
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Vertex {
    pub position: Vector3,
    #[serde(default = "default_inv_mass")]
    pub inv_mass: f32,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VertexId(pub u32);

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Tetrahedron {
    pub indices: [VertexId; 4],
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Edge(pub VertexId, pub VertexId);

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Triangle([VertexId; 3]);

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Struct to contain data of a delanuay tetrahedralized mesh.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Tetrahedral {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub tetrahedra: Vec<Tetrahedron>,
    pub faces: Vec<Triangle>,
}

impl Tetrahedral {
    /// Parse a generic tetgen file format
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

    fn parse_indices(tokens: &[&str], start: usize, count: usize) -> Result<Vec<u32>> {
        tokens[start..start + count]
            .iter()
            .map(|&t| t.parse().map_err(Into::into))
            .collect()
    }

    /// Translate the entire mesh by a vector.
    pub fn translate(&mut self, by: Vector3) {
        for vertex in &mut self.vertices {
            vertex.position += by;
        }
    }

    pub fn from_files(prefix: &str) -> Result<Self> {
        let vertices = Self::parse_file(&format!("{prefix}.node"), |tokens| {
            let coords: Vec<f32> = tokens[1..4]
                .iter()
                .map(|&t| t.parse().map_err(Into::into))
                .collect::<Result<_>>()?;
            Ok(Vertex {
                position: Vector3::new(coords[0], coords[1], coords[2]),
                inv_mass: 1.0,
            })
        })?;

        let edges = if Path::new(&format!("{prefix}.edge")).exists() {
            Self::parse_file(&format!("{prefix}.edge"), |tokens| {
                let ids = Self::parse_indices(tokens, 1, 2)?;
                Ok(Edge(VertexId(ids[0]), VertexId(ids[1])))
            })?
        } else {
            Vec::new()
        };

        let faces = if Path::new(&format!("{prefix}.face")).exists() {
            Self::parse_file(&format!("{prefix}.face"), |tokens| {
                let ids = Self::parse_indices(tokens, 1, 3)?;
                Ok(Triangle([
                    VertexId(ids[0]),
                    VertexId(ids[1]),
                    VertexId(ids[2]),
                ]))
            })?
        } else {
            Vec::new()
        };

        let tetrahedra = if Path::new(&format!("{prefix}.ele")).exists() {
            Self::parse_file(&format!("{prefix}.ele"), |tokens| {
                let ids = Self::parse_indices(tokens, 1, 4)?;
                Ok(Tetrahedron {
                    indices: [
                        VertexId(ids[0]),
                        VertexId(ids[1]),
                        VertexId(ids[2]),
                        VertexId(ids[3]),
                    ],
                })
            })?
        } else {
            Vec::new()
        };

        Ok(Self {
            vertices,
            edges,
            tetrahedra,
            faces,
        })
    }

    /// Load tetrahedral mesh from bincode file
    pub fn from_bincode(filename: &str) -> Result<Self> {
        let data = std::fs::read(filename)?;
        let mesh: Self = bincode::deserialize(&data)?;
        Ok(mesh)
    }

    /// Export mesh to bincode format
    pub fn export_to_bincode(&self, output_path: &str) -> Result<()> {
        use std::io::Write;
        use tracing::{debug, info};

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

    /// Load mesh with automatic format detection
    pub fn load_mesh(mesh_path: &str) -> Result<Self> {
        use tracing::{debug, error, info};

        info!(mesh_path, "Attempting to load mesh");

        let mesh = if std::path::Path::new(mesh_path)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
        {
            debug!("Loading as bincode file");
            Self::from_bincode(mesh_path)
        } else {
            debug!("Loading as tetgen files");
            Self::from_files(mesh_path)
        };

        match &mesh {
            Ok(m) => {
                info!(
                    vertices = m.vertices.len(),
                    edges = m.edges.len(),
                    faces = m.faces.len(),
                    tetrahedra = m.tetrahedra.len(),
                    "Mesh loaded successfully"
                );
            }
            Err(_) => {
                error!(mesh_path, "Failed to load mesh");
            }
        }

        mesh
    }

    /// Draw wireframe of the mesh
    pub fn draw_wireframe(&self, d3: &mut RaylibMode3D<RaylibDrawHandle>, color: Color) {
        // Draw explicit edges if available
        for edge in &self.edges {
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

    /// Get bounding box of the mesh
    pub fn bounding_box(&self) -> (Vector3, Vector3) {
        if self.vertices.is_empty() {
            return (Vector3::zero(), Vector3::zero());
        }

        let mut min = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for vertex in &self.vertices {
            min.x = min.x.min(vertex.position.x);
            min.y = min.y.min(vertex.position.y);
            min.z = min.z.min(vertex.position.z);
            max.x = max.x.max(vertex.position.x);
            max.y = max.y.max(vertex.position.y);
            max.z = max.z.max(vertex.position.z);
        }

        (min, max)
    }

    /// Draw filled faces
    pub fn draw_faces(&self, d3: &mut RaylibMode3D<RaylibDrawHandle>, color: Color) {
        for face in &self.faces {
            let verts = [
                self.vertices.get((face.0[0].0 - 1) as usize),
                self.vertices.get((face.0[1].0 - 1) as usize),
                self.vertices.get((face.0[2].0 - 1) as usize),
            ];
            if let [Some(v1), Some(v2), Some(v3)] = verts {
                let p1 = v1.position;
                let p2 = v2.position;
                let p3 = v3.position;
                d3.draw_triangle3D(p1, p2, p3, color);
            }
        }
    }
}

impl Index<VertexId> for Vec<Vertex> {
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

impl IndexMut<VertexId> for Vec<Vertex> {
    fn index_mut(&mut self, index: VertexId) -> &mut Self::Output {
        let len = self.len();
        self.get_mut((index.0 - 1) as usize)
            .unwrap_or_else(|| panic!("Invalid vertex id: {}, only {} available.", index.0, len))
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
        assert_eq!(mesh.edges.len(), 1);
        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.tetrahedra.len(), 1);

        // Cleanup
        for ext in &["node", "edge", "face", "ele"] {
            let _ = fs::remove_file(format!("{prefix}.{ext}"));
        }
    }
}
