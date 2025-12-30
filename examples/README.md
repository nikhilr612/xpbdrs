# XPBD Examples

## Cloth Simulation

Demonstrates cloth physics using a triangulated surface mesh on a subdivided square. The cloth is fixed at boundaries and after some time a force is applied to its center.

### Usage

```bash
cargo run --example cloth
cargo run --example cloth -- --resolution 30 --size 6.0
```

### Controls

- **X**: Toggle Wireframe | **F**: Toggle Faces | **R**: Reset
- **Space/Shift**: Camera up/down | **Mouse**: Rotate camera

### How it Works

Creates a subdivided square mesh with triangular faces. Boundary vertices are fixed in place with very small inverse mass, while interior vertices can move freely. Edge constraints maintain structural integrity, and weak bending constraints prevent excessive folding.

After 3 seconds, a strong downward force is applied at the cloth center for 1 second.

### Parameters

- `--resolution`: Grid resolution (default: 20)
- `--size`: Cloth size in world units (default: 5.0)

## Inflation Demo

Demonstrates soft body inflation using `p_volume` to scale tetrahedral mesh volumes. Spawns a `deci_spot` mesh at height and inflates it over time.

### Usage

```bash
cargo run --example inflation_demo
cargo run --example inflation_demo -- path/to/custom/mesh
```

### Controls

- **X**: Wireframe | **F**: Faces | **P**: Pause | **R**: Reset
- **Space/Shift**: Camera up/down | **Mouse**: Rotate camera

### How it Works

The mesh falls for 2 seconds, then inflates over 5 seconds by scaling `p_volume` from 1.0x to 2.5x in `XpbdParams`. This creates internal pressure while edge constraints maintain structural integrity.

Uses tetrahedral meshes (tetgen `.node/.ele/.edge/.face` files or `.bin` format). The default `deci_spot.bin` is a decimated variant of Spot the Cow by Keenan Crane.
