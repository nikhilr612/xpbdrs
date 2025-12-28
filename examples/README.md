# XPBD Examples

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
