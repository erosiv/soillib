# soillib

C++20 library and toolbox for procedural geomorphology simluation

Based on concepts developed by and maintained by [Nicholas McDonald](https://github.com/weigert).

<p align="center">
<img alt="Heightmap with Normal Shading" src="https://github.com/erosiv/soillib/assets/6532707/51cd9ef0-13a1-4831-8e12-0806fad1580d" width="75%" align="center"/>
</p>

Normal Shaded Heightmap, 3D View (see: `tools/view`)

<p align="center">
<img alt="Relief Shaded Heightmap with Aerial Perspective" src="https://github.com/erosiv/soillib/assets/6532707/b6bd03eb-ccb5-4287-8499-5948895652ac" width="75%" align="center"/>
</p>

Relief Shaded Heightmap with Aerial Perspective (see: `tools/map_relief`)

<p align="center">
<img alt="Erosion Normal Map with Non-Regular Map Shape" src="https://github.com/erosiv/soillib/assets/6532707/737d417d-a308-48cd-94f4-eef43a02bd1b" width="75%" align="center"/>
</p>

Irregular Map Shape Normal Map (see: `tools/quad_hydrology`)

## Description

`soillib` is a unified C++20 library for procedural geomorphology simulation.

`soillib` provides modularized and unified concepts for many aspects of geomorphological simulations, from high-performance map structures to unified particle physics and a unified data import and export interface.

Overall, this allows for creating complex geomorphological simulations through a thin implementation layer with predictable results and without requiring re-implementation. Examples can be found in the `tools` directory.

`soillib` is inspired by a number of predecessor systems and the difficulty of maintaining them all at the same time as concepts evolve. This allows for the maintenance of a single library, and re-implementing these programs on top of this library.

#### Highlights

- Unified cell-pool templates for efficient memory management
- Index templated map structures w. iterators for self-defined memory layouts
- Unified image import / export, including floating-point `.tiff` data. Saving and loading maps is extremely trivial
- Configurability of all core data-structures using `yaml`. No need to recompile, just update a single configuration file to change all simulation parameters. This is functional but still WIP and will be improved more in the futur.

This allows you to do things like e.g. using a morton-order indexed map with a fixed memory chunk, so that you can operate on contiguous segments which are spatially separated in parallel.

#### Why C++20?

Concepts and type constraints are extremely convenient for defining complex operations which can be implemented for certain map and cell types, without becoming too specific.

As an example, see `soillib/model/surface.hpp`.

## Utilization

Install the library headers using the `Makefile`.

```bash
make all
```
