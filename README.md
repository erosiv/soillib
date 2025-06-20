# soillib

soillib is a library and toolbox for numerical geomorphology simulation on the GPU

Written in in C++23 + CUDA and exposed to Python3 through bindings.

Maintained by [Erosiv](https://erosiv.studio). Based on concepts developed by and maintained by [Nicholas McDonald](https://github.com/weigert).

<p align="center">
<img alt="Terrain Generated with Soillib" src="https://github.com/user-attachments/assets/74a6d55f-2f1c-43f7-b911-81ac91a1374b" width="75%" align="center"/>
</p>

## Description

`soillib` is a unified C++23 library for numerical geomorphology simulation, with a Python3 package layer built on top. The library is designed with a high degree of compatiblity with Python in mind.

`soillib` provides modularized and unified concepts for many aspects of geomorphological simulations, from high-performance indexing structures to unified particle physics and unified data import and export interfaces, including native `GeoTIFF` support.

`soillib` is fully statically typed in C++, while allowing for dynamic types in the python interface. It achieves this through a selector pattern, which generates statically typed code for all permitted types (constrained by concepts), while choosing these paths dynamically. This leads to deep inlining and full static performance benefits.

This allows for creating complex geomorphological simulations through elegant modular concepts in Python. All examples are implemented in Python, but are equally reproducible in C++.

`soillib` is interoperable with popular Python packages like numpy and pytorch, making it easy to integrate into new or existing projects quickly. 

`soillib` is inspired by a number of predecessor systems and the difficulty of maintaining them all at the same time as concepts evolve. This allows for the maintenance of a single library, and re-implementing these programs on top of this library easily.

#### Features / Highlights

- GPU First Kernelized Erosion Models
- A Library of Kernelized Operations for Numerical Geomorphology
- Fully Statically Typed C++23 Library with Concepts
- Dynamically Typed Python Module
- Interoperability with Numpy / PyTorch
- Unified image import / export, including floating-point `.tiff` data and native GeoTIFF support.

#### Why C++23?

Concepts and type constraints are extremely convenient for defining complex operations which can be implemented for certain map and cell types, without becoming too specific.

Additionally, the introduction of "deducing this" in `C++23` as well as the convenient `std::format` and `std::print` are features that reduce design complexity of the library.

## Utilization

### Installing for Python

Install through [pypi.org](https://pypi.org/project/soillib/) using `pip`:

```bash
pip install soillib
```

Note: Currently only Linux builds are supported. Other builds should come online shortly.

### Building from Source

`soillib` is a combination of CUDA code compiled with `nvcc`, regular C++ code compiled with `g++` and a python module that requires linking with `nanobind`. Additionally, it depends on `glm` and `libtiff`.

#### Dependencies

Make sure to instal the cuda toolkit so that you have the `nvcc` compiler, the cuda runtime headers and the cuda library.

- GLM
- LibTIFF
- CUDA Toolkit
- CMake
- Python3
- 

##### Ubuntu

```bash
sudo apt-get install libglm-dev
```

##### Fedora

```bash
sudo dnf install glm-devel
```

##### Building with CMake

Update the git submodule to install nanobind.

```bash
git submodule update --init --recursive
```

Build nanobind and soillib as a C++ library using CMake.

```bash
cmake -S . -B build
cmake --build build -j1
cmake --install build
```

Note that installation with cmake might require super-user privilege.

Build the python bindings using make:

```bash
make python
```

Note that this will install the headers, compile the python shared object, build a `.whl` file and install it with `pip`. Inspect the `Makefile` for more granular control.

Note that the library is built with `nvcc`. Linking the library with your project does not require using `nvcc` and can be done with `g++`. 

## Windows

- Install CMake
- Install CUDA Runtime and Toolkit 12.9
