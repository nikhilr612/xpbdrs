# XPBD Examples

## Cloth Simulation

Demonstrates cloth physics using a triangulated surface mesh on a subdivided square. The cloth is fixed at boundaries and after some time a force is applied to its center.

### Usage

```bash
cargo run --example deform_demo
cargo run --example deform_demo -- --resolution 30 --size 6.0
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

## Cloth Draping

Demonstrates realistic cloth draping simulation where a cloth mesh falls and drapes over a sphere. All cloth vertices have uniform mass, creating natural cloth behavior under gravity.

### Usage

```bash
cargo run --example cloth_draping
cargo run --example cloth_draping -- --resolution 25 --size 4.0 --height 2.5 --sphere-radius 1.0
```

### Controls

- **X**: Toggle Wireframe | **F**: Toggle Faces | **R**: Reset
- **Space/Shift**: Camera up/down | **Mouse**: Rotate camera

### How it Works

Creates a subdivided square cloth mesh with all vertices having identical mass. The cloth is spawned at a specified height above a sphere and falls under gravity. Collision detection ensures the cloth drapes naturally over the sphere rather than passing through it.

The simulation uses sphere collision handling in each physics substep and applies damping to create realistic cloth movement.

### Parameters

- `--resolution`: Cloth grid resolution (default: 25)
- `--size`: Cloth size in world units (default: 4.0)
- `--height`: Spawn height above sphere (default: 2.5)
- `--sphere-radius`: Radius of the draping sphere (default: 1.0)

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

## Spot in a Box

Demonstrates physics simulation with box collision constraints and periodic force application. Spawns a `deci_spot` mesh at height with gravity, constrains it within a box using collision callbacks, and periodically applies preset forces at the centroid for short durations.

### Usage

```bash
cargo run --example spot_in_box
```

### Controls

- **R**: Reset | **Mouse**: Rotate camera

### How it Works

The mesh spawns at a fixed height and falls under gravity within a box constraint. Every 1.5 seconds, a preset force vector from a cycling list is applied for 0.1 seconds using an inverse square law (F = k/r²) where force magnitude is inversely proportional to the square of distance from the mesh centroid. Forces below a threshold distance are ignored, and maximum force is capped for stability.

The simulation cycles through 4 different preset force directions, creating predictable but varied motion as the mesh bounces around the box.
