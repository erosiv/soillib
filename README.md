# soillib

soillib is a library and toolbox for procedural geomorphology simulation.

Written in in C++23 and exposed to Python3 through bindings.

Based on concepts developed by and maintained by [Nicholas McDonald](https://github.com/weigert).

<p align="center">
<img alt="Heightmap with Normal Shading" src="https://github.com/erosiv/soillib/assets/6532707/51cd9ef0-13a1-4831-8e12-0806fad1580d" width="75%" align="center"/>
</p>

Normal Shaded Heightmap, 3D View

<p align="center">
<img alt="Relief Shaded Heightmap with Aerial Perspective" src="https://github.com/erosiv/soillib/assets/6532707/b6bd03eb-ccb5-4287-8499-5948895652ac" width="75%" align="center"/>
</p>

Relief Shaded Heightmap with Aerial Perspective

<p align="center">
<img alt="Erosion Normal Map with Non-Regular Map Shape" src="https://github.com/erosiv/soillib/assets/6532707/737d417d-a308-48cd-94f4-eef43a02bd1b" width="75%" align="center"/>
</p>

Irregular Map Shape Normal Map

<p align="center">
<img alt="Erosion Normal Map with Non-Regular Map Shape" src="https://github.com/erosiv/soillib/assets/6532707/86b7e873-bd4c-4035-8b8e-fe37dde62a71" width="75%" align="center"/>
</p>

Binary Soil-Mixture Transport Albedo Map

## Description

`soillib` is a unified C++23 library for procedural geomorphology simulation, with a Python3 package layer built on top. The library is designed with a high degree of compatiblity with Python in mind. 

`soillib` provides modularized and unified concepts for many aspects of geomorphological simulations, from high-performance indexing structures to unified particle physics and unified data import and export interfaces, including native `GeoTIFF` support.

This allows for creating complex geomorphological simulations through elegant modular concepts in Python. All examples are implemented in Python, but are equally reproducible in C++.

`soillib` is interoperable with popular Python packages like numpy, making it easy to integrate into new or existing projects quickly. 

`soillib` is inspired by a number of predecessor systems and the difficulty of maintaining them all at the same time as concepts evolve. This allows for the maintenance of a single library, and re-implementing these programs on top of this library easily.

#### Features / Highlights

- Generic index generators for arbitrarily shaped maps
- Unified memory pool buffers for efficient memory management
- Unified image import / export, including floating-point `.tiff` data and native GeoTIFF support.
- Generic matrix types for modelling sediment dynamics with different characteristic properties
- Composable nodes for user-defined property dependencies
- Underlying types are fully strict-typed
- Polymorphic python type deduction leads to strict-typed call paths for deep inlining

#### Why C++23?

Concepts and type constraints are extremely convenient for defining complex operations which can be implemented for certain map and cell types, without becoming too specific.

Additionally, the introduction of "deducing this" in `C++23` as well as the convenient `std::format` and `std::print` are features that reduce design complexity of the library.

## Utilization

### Python

Install through [pypi.org](https://pypi.org/project/soillib/) using `pip`:

```bash
pip install soillib
```

Note: Currently only Linux builds are supported. Other builds should come online shortly.

### C++

Install the library headers using the `Makefile`:

```bash
make source
```

### Building from Source

The library headers can be installed and the python package built and installed using the `Makefile`:

```bash
make python
```

Note that this will install the headers, compile the python shared object, build a `.whl` file and install it with `pip`. Inspect the `Makefile` for more granular control.

This requires a working installation of `nanobind`. Note that the building code has not been made fully platform independent yet - some effort is needed to build on alternative platforms by modifying the parameters in the `Makefiles`.

## ToDo

This is a list of changes I would like to integrate into `soillib`. If you are reading this and motivated to implement any of these, I would be happy to accept any pull requests.

- Make the enum switch based polymorphism more generic / nicely structured in the code
- Lazy Node Evaluation w. Deep Inlining
	- Similar to `pytorch`, figure out the most elegant way to make the node evaluation lazy and efficient.
- Node caching / baking
	- Introduce the ability to "cache" node data, for instance when it is known that the values in a map will be sampled randomly multiple times to reduce compute effort.
- Re-Interpretable Buffer Types
	- Make buffers more generic and re-interpretable as strict-typed views. Use this to eliminate the concept of a vector-value buffer, just having the value and viewing it as a vector type.
- Implement all remaining matrix types from the C++ library.
- Re-introduce the wind particle.
- Re-introduce layer maps / stratigraphy.