# soillib

soillib is a library and toolbox for numerical geomorphology simulation on the GPU

Written in in C++23 + CUDA and exposed to Python3 through bindings.

Maintained by [Erosiv](https://erosiv.studio). Based on concepts developed by and maintained by [Nicholas McDonald](https://github.com/weigert).

<p align="center">
<img alt="Terrain Generated with Soillib" src="https://github.com/erosiv/soillib/image/render.png" width="75%" align="center"/>
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

`soillib` is a combination of CUDA code compiled with `nvcc`, regular C++ code compiled with `g++` and a python module that requires linking with `nanobind`. The compilation process is currently slightly more involved than is desirable.

##### Building soillib

`soillib` consists of header files `<soillib/*>` and a library `libsoil.a`. To install the header files and the library, build with the `Makefile`:

```bash
make source
```

Note that the library is build with `nvcc`. Linking the library with your project does not require using `nvcc` and can be done with `g++`. 

##### Building the Python Module

With the header files and library installed, the python library can be built using the `Makefile`:

```bash
make python
```

This requires a working installation of `nanobind`. `nanobind` has to be built with position independent code, as it is simply linked in this step. Note that the building code has not been made fully platform independent yet - some effort is needed to build on alternative platforms by modifying the parameters in the `Makefiles`.

Note that this will install the headers, compile the python shared object, build a `.whl` file and install it with `pip`. Inspect the `Makefile` for more granular control.

#### Building with CUDA Enabled

The build will automatically detect whether you have CUDA installed:

```bash
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo
sudo dnf clean all
sudo dnf module disable nvidia-driver
sudo dnf -y install cuda
```