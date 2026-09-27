cmake_minimum_required(VERSION 3.21.0)

#
# Version
#

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/VERSION" SOIL_VERSION LIMIT_COUNT 1)

#
# Project Configuration
#

set(PROJECT_NAME soillib)
project(${PROJECT_NAME} VERSION ${SOIL_VERSION} LANGUAGES C CXX CUDA)

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

add_subdirectory(source)  # C++/CUDA library

if(PROJECT_IS_TOP_LEVEL)
  add_subdirectory(python)
endif()