# soillib

C++20 library and toolbox for procedural geomorphology simluation

Based on concepts developed by and maintained by [Nicholas McDonald](https://github.com/weigert).

<p align="center">
<img alt="Heightmap with Normal Shading" src="https://github.com/erosiv/soillib/assets/6532707/51cd9ef0-13a1-4831-8e12-0806fad1580d" width="75%" align="center"/>
</p>

Normal Shaded Heightmap, 3D View (see: `tool/viewer`)

<p align="center">
<img alt="Relief Shaded Heightmap with Aerial Perspective" src="https://github.com/erosiv/soillib/assets/6532707/b6bd03eb-ccb5-4287-8499-5948895652ac" width="75%" align="center"/>
</p>

Relief Shaded Heightmap with Aerial Perspective (see: `tool/relief`)

<p align="center">
<img alt="Erosion Normal Map with Non-Regular Map Shape" src="https://github.com/erosiv/soillib/assets/6532707/737d417d-a308-48cd-94f4-eef43a02bd1b" width="75%" align="center"/>
</p>

Irregular Map Shape Normal Map (see: `model/quad_hydrology`)

<p align="center">
<img alt="Erosion Normal Map with Non-Regular Map Shape" src="https://github.com/erosiv/soillib/assets/6532707/86b7e873-bd4c-4035-8b8e-fe37dde62a71" width="75%" align="center"/>
</p>

Binary Soil-Mixture Transport Albedo Map (see: `model/basic_hydrology_mixture`)

## Description

`soillib` is a unified C++20 library for procedural geomorphology simulation.

`soillib` provides modularized and unified concepts for many aspects of geomorphological simulations, from high-performance map structures to unified particle physics and a unified data import and export interfaces, including native `GeoTIFF` support.

Overall, this allows for creating complex geomorphological simulations through a thin implementation layer with predictable results and without requiring re-implementation. Example implementations can be found in the `model` directory. Utility programs for the manipulation of data can be found in the `tool` directory.

`soillib` is inspired by a number of predecessor systems and the difficulty of maintaining them all at the same time as concepts evolve. This allows for the maintenance of a single library, and re-implementing these programs on top of this library.

#### Highlights

- Unified cell-pool templates for efficient memory management
- Index templated map structures w. iterators for self-defined memory layouts
- Unified image import / export, including floating-point `.tiff` data. Saving and loading maps is extremely trivial
- Configurability of all core data-structures using `yaml`. No need to recompile, just update a single configuration file to change all simulation parameters. This is functional but still WIP and will be improved more in the futur.

This allows you to do things like e.g. using a morton-order indexed map with a fixed memory chunk, so that you can operate on contiguous segments which are spatially separated in parallel.

#### Tools and Models

| Tool          | Description |
| ----          | ----------- |
| `tool/viewer` | View a floating-point `.tiff` height-map (or directory) as a shaded 3D model. Note that for a directory, they must have `GeoTIFF` metadata for relative positioning. |
| `tool/relief` | Render a relief-shaded `.png` for a single `.tiff` height-map.

| Model                   | Description |
| -----                   | ----------- |
| `model/basic_hydrology` | Basic rectangular map hydrology model for meandering river erosion simulation. |

#### Why C++20?

Concepts and type constraints are extremely convenient for defining complex operations which can be implemented for certain map and cell types, without becoming too specific.

As an example, see `soillib/model/surface.hpp`.

## Utilization

Install the library headers using the `Makefile`.

```bash
make all
```

##### ToDo

This is a list of changes I would like to integrate into `soillib`. If you are reading this and motivated to implement any of these, I would be happy to accept any pull requests.

- `soillib/io`:
	- Add a convenience image iterator
	- Strict-Typed (Templated) Image Types w. Run-Time Deduction after Image Loading and Bit-Depth Determination, using e.g. `std::variant`. E.g. determine whether a `.tiff` is `float32` or `float64`, but yield a strict-type.
	- Pre-Determined templated image types
	- Convenient but decoupled interoperability w. OpenGL textures through `TinyEngine` somehow.
	- Better image reallocation handling w. alignment and bit-masks.
	- Unified "GDAL No Data" point handling in `GeoTIFF`, including a meshing concept.