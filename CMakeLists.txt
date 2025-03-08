cmake_minimum_required(VERSION 3.14)
project(soillib LANGUAGES CXX CUDA)

include(FetchContent)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find Python3 with Interpreter and Development.Module
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module Development)
if(NOT EXISTS "${Python3_INCLUDE_DIRS}/Python.h")
  message(FATAL_ERROR "Python.h not found in ${Python3_INCLUDE_DIRS}. Install python3.12-dev.")
endif()
if(NOT Python3_LIBRARIES)
  message(WARNING "Python3_LIBRARIES is empty. Manually setting it.")
  set(Python3_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpython3.12.so")
endif()

# Find CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  message(STATUS "CUDA found: ${CUDA_VERSION} at ${CUDA_TOOLKIT_ROOT_DIR}")
else()
  message(FATAL_ERROR "CUDA not found. Please install CUDA.")
endif()

# Fetch dependencies
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(nanobind)

FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

# Find TIFF library
find_package(TIFF REQUIRED)

# Create a temporary include directory in build to map "soillib/"
set(SOILLIB_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${SOILLIB_INCLUDE_DIR})

# Remove existing soillib directory if it exists, then create symlink
if(EXISTS "${SOILLIB_INCLUDE_DIR}/soillib")
  file(REMOVE_RECURSE ${SOILLIB_INCLUDE_DIR}/soillib)
endif()
execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${CMAKE_SOURCE_DIR}/source ${SOILLIB_INCLUDE_DIR}/soillib
  RESULT_VARIABLE SYMLINK_RESULT
)
if(NOT SYMLINK_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to create symlink: ${SYMLINK_RESULT}")
endif()

# Include directories
include_directories(
  ${SOILLIB_INCLUDE_DIR}  # For soillib headers
  ${CUDA_INCLUDE_DIRS}    # For cuda_runtime.h
  ${Python3_INCLUDE_DIRS}
  ${nanobind_SOURCE_DIR}/include
  ${nanobind_SOURCE_DIR}/ext/robin_map/include
  ${glm_SOURCE_DIR}
)

# Define Python module with CUDA sources, excluding erosion_thermal.cu to avoid duplicates
nanobind_add_module(
  soillib
  python/source/soillib.cpp
  python/source/index.cpp
  python/source/io.cpp
  python/source/op.cpp
  python/source/buffer.cpp
  python/source/util.cpp
  ${CMAKE_SOURCE_DIR}/source/core/buffer.cu
  ${CMAKE_SOURCE_DIR}/source/op/common.cu
  ${CMAKE_SOURCE_DIR}/source/op/erosion.cu
  # ${CMAKE_SOURCE_DIR}/source/op/erosion_thermal.cu # gives issues with duplicate symbols
  ${CMAKE_SOURCE_DIR}/source/op/flow.cu
  ${CMAKE_SOURCE_DIR}/source/op/math.cu
)

# Set CUDA architectures (no separable compilation needed)
set_target_properties(soillib PROPERTIES
  CUDA_ARCHITECTURES "50;60;70;75;80"  # Supports Maxwell (5.0), Pascal (6.0), Volta (7.0), Turing (7.5), Ampere (8.0)
)

# Add CUDA-specific compile options
target_compile_options(soillib PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -Xcudafe=--diag_suppress=177
    -Xcudafe=--diag_suppress=445
    -Xcudafe=--diag_suppress=2361
    -Xcudafe=--diag_suppress=20011
    -Xcudafe=--diag_suppress=20012
    -Wno-deprecated-gpu-targets
  >
)

# Ensure the compiler can find headers
target_include_directories(soillib PRIVATE
  ${SOILLIB_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}    # For cuda_runtime.h
  ${Python3_INCLUDE_DIRS}
  ${nanobind_SOURCE_DIR}/include
  ${nanobind_SOURCE_DIR}/ext/robin_map/include
  ${glm_SOURCE_DIR}
)

# Link against libraries
target_link_libraries(soillib PRIVATE glm TIFF::TIFF ${Python3_LIBRARIES} ${CUDA_LIBRARIES})

# Detect Python site-packages directory within the virtualenv
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE PYTHON_VENV_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Installing to virtualenv site-packages: ${PYTHON_VENV_SITE_PACKAGES}")

install(
  TARGETS soillib
  LIBRARY DESTINATION ${PYTHON_VENV_SITE_PACKAGES}
)

# Debugging output
message(STATUS "SOILLIB include directory: ${SOILLIB_INCLUDE_DIR}")
message(STATUS "CUDA include directory: ${CUDA_INCLUDE_DIRS}")
message(STATUS "Symlink created at: ${SOILLIB_INCLUDE_DIR}/soillib -> ${CMAKE_SOURCE_DIR}/source")
message(STATUS "nanobind include directory: ${nanobind_SOURCE_DIR}/include")
message(STATUS "robin_map include directory: ${nanobind_SOURCE_DIR}/ext/robin_map/include")
message(STATUS "Python include directory: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python3_LIBRARIES}")
message(STATUS "glm include directory: ${glm_SOURCE_DIR}")