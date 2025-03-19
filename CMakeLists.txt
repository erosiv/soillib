# soillib CMake Build Configuration
cmake_minimum_required(VERSION 3.3)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-13")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80") # Supports Maxwell (5.0), Pascal (6.0), Volta (7.0), Turing (7.5), Ampere (8.0)

project(soillib LANGUAGES CXX CUDA)

add_subdirectory(ext)     # Build Nanobind
add_subdirectory(source)  # Build Soillib
add_subdirectory(python)  # Build Python Bindings