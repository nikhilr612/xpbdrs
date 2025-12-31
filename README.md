# Xpbdrs

A Rust implementation of Extended Position Based Dynamics (XPBD) for real-time simulation of deformable bodies using tetrahedral and triangulated surface meshes.
XPBDRS provides a constraint-based physics engine implementing the Extended Position Based Dynamics algorithm which extends the classical Position Based Dynamics framework by incorporating material compliance parameters through Lagrange multiplier accumulation enabling compliance-based material modeling. The library supports multi-mesh representations, adaptive constraint optimization, and real-time visualization for interactive simulation of soft bodies, cloth dynamics, and volumetric deformation.

### XPBD Algorithm

The XPBD method solves constraint systems by accumulating Lagrange multipliers over substeps, enabling proper material compliance modeling:

```
For each substep Δt:
  1. Predict positions: x* = x + v·Δt + a·Δt²
  2. For each constraint C(x):
     - Compute constraint violation δC = C(x*)
     - Calculate compliance α = compliance / Δt²
     - Update multiplier: Δλ = -δC / (∇C^T M^(-1) ∇C + α)
     - Apply position correction: Δx = M^(-1) ∇C Δλ
  3. Update velocities: v = (x_new - x_old) / Δt
```

### Constraint Types

**Edge Length Constraints**: Maintain structural integrity of mesh connectivity
- Constraint function: C(x) = |x₁ - x₂| - L₀
- Gradient: ∇C = (x₁ - x₂) / |x₁ - x₂|

**Tetrahedral Volume Constraints**: Preserve volumetric properties for soft body simulation  
- Constraint function: C(x) = V(x₁,x₂,x₃,x₄) - V₀
- Volume scaling parameter p_volume enables controlled inflation/deflation

**Weak Bending Constraints**: Prevent excessive folding in surface meshes
- Applied to adjacent triangle pairs sharing an edge
- Maintains surface curvature characteristics

**Substep Integration**: Configurable substep count enables trade-offs between accuracy and performance based on application requirements.

## File Format Support

- **TetGen ASCII**: Standard `.node`, `.ele`, `.edge`, `.face` format compatibility
- **Binary Serialization**: Optimized `.bin` format using serde/bincode for reduced I/O overhead
- **Mesh Conversion**: Bidirectional conversion between ASCII and binary representations

## Usage

```rust
use xpbdrs::{
    mesh::Tetrahedral,
    xpbd::{XpbdParams, XpbdState, step_basic}
};

// Load tetrahedral mesh from TetGen files
let mut mesh = Tetrahedral::from_files("mesh_prefix")?;
let initial_values = mesh.constraints.evaluate(&mesh.vertices);

// Configure simulation parameters
let params = XpbdParams {
    length_compliance: 0.001,      // Material stiffness
    volume_compliance: 0.001,      // Volume preservation
    n_substeps: 10,                // Stability vs performance
    time_substep: 0.016 / 10.0,    // 60 FPS with 10 substeps
    ..Default::default()
};

// Initialize simulation state
let mut state = XpbdState::new(
    mesh.vertices.len(), 
    mesh.constraints.size()
);

// Simulation loop with ground collision
loop {
    state = step_basic(
        &params, 
        state, 
        &mut mesh, 
        &initial_values,
        |v| v.position.y = v.position.y.max(0.0) // Ground plane at y=0
    );
}
```

### Command Line Interface

```bash
# Convert TetGen mesh to optimized binary format
cargo run -- export -i mesh_prefix -o output.bin

# Launch demo softbody viwer (requires raylib feature)
cargo run --features raylib -- demo mesh.bin

# Run example simulations (require raylib feature for visualization)
cargo run --features raylib --example cloth_draping
cargo run --features raylib --example inflation_demo  
cargo run --features raylib --example spot_in_box
```

## Examples

The `examples/` directory contains comprehensive demonstrations of library capabilities:

- **Cloth Draping**: Realistic cloth simulation with sphere collision detection
- **Surface Deformation**: Interactive plane mesh with localized force application  
- **Inflation Demo**: Soft body volume scaling using tetrahedral constraints
- **Spot in Box**: Constrained dynamics with periodic force impulses

Each example includes parameter customization, interactive controls, and visual output modes. All examples require the `raylib` feature to be enabled for 3D visualization.

**Note**: The core XPBD simulation library can be used without raylib for headless physics computation. Only the examples, demos, and visualization components require the raylib feature.

## Dependencies

- **glam**: Vector and matrix mathematics with serde support
- **serde/bincode**: Mesh serialization  
- **tracing**: Structured logging
- **clap**: Command-line argument parsing (dev dependency)

### Optional Features

- **raylib** (optional): 3D rendering and user interaction for examples and demos
  - Enable with `cargo run --features raylib` or `default-features = false` in Cargo.toml
