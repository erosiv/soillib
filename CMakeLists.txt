cmake_minimum_required(VERSION 3.14)
project(soillib LANGUAGES CXX)

include(FetchContent)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Find Python3 with Interpreter and Development.Module
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module Development)
if(NOT EXISTS "${Python3_INCLUDE_DIRS}/Python.h")
  message(FATAL_ERROR "Python.h not found in ${Python3_INCLUDE_DIRS}. Install python3.12-dev.")
endif()
if(NOT Python3_LIBRARIES)
  message(WARNING "Python3_LIBRARIES is empty. Manually setting it.")
  set(Python3_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpython3.12.so")
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

# Include directories
include_directories(${Python3_INCLUDE_DIRS})

# Define Python module
nanobind_add_module(
  soillib
  python/source/soillib.cpp
  python/source/index.cpp
  python/source/io.cpp
  python/source/matrix.cpp
  python/source/model.cpp
  python/source/node.cpp
  python/source/particle.cpp
  python/source/util.cpp
)

# Ensure the compiler can find headers
target_include_directories(soillib PRIVATE
  ${CMAKE_SOURCE_DIR}/source
  ${CMAKE_SOURCE_DIR}/python/source
  ${nanobind_SOURCE_DIR}/include
  ${nanobind_SOURCE_DIR}/ext/robin_map/include
  ${Python3_INCLUDE_DIRS}
  ${glm_SOURCE_DIR}
)

# Link against glm, TIFF, and Python libraries
target_link_libraries(soillib PRIVATE glm TIFF::TIFF ${Python3_LIBRARIES})

# Detect python user site-packages directory
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getusersitepackages())"
  OUTPUT_VARIABLE PYTHON_USER_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Installing to User's Python site-packages: ${PYTHON_USER_SITE_PACKAGES}")

install(
  TARGETS soillib
  LIBRARY DESTINATION ${PYTHON_USER_SITE_PACKAGES}/
)

# Debugging output
message(STATUS "nanobind include directory: ${nanobind_SOURCE_DIR}/include")
message(STATUS "robin_map include directory: ${nanobind_SOURCE_DIR}/ext/robin_map/include")
message(STATUS "Python include directory: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python3_LIBRARIES}")
message(STATUS "glm include directory: ${glm_SOURCE_DIR}")
