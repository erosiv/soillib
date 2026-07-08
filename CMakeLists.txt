cmake_minimum_required(VERSION 3.21.0)

#
# Project Configuration
#

set(PROJECT_NAME soillib)
set(PROJECT_VERSION 1.1.2)

project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89 90 100 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

#
# Project Components
#

add_subdirectory(source)  # Build Library
if(PROJECT_IS_TOP_LEVEL)
  add_subdirectory(python)  # Build Bindings
endif()