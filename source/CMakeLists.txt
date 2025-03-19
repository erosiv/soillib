#
# soillib C++ Library Target
#

add_library(soil STATIC)
set_target_properties(soil PROPERTIES 
  LINKER_LANGUAGE CUDA
)
set(CMAKE_CUDA_FLAGS "-diag-suppress 20012 -diag-suppress 20013 -diag-suppress 20015 -diag-suppress 445 -fPIC -c -O3")

# Dependency Wrangling

find_path(GLM_INCLUDE_DIR glm/glm.hpp)
if(GLM_INCLUDE_DIR)
  message(STATUS "GLM Install Location: ${GLM_INCLUDE_DIR}")
  include_directories(${GLM_INCLUDE_DIR})
else()
  message(FATAL_ERROR "glm is not installed. Please install glm.")
endif(GLM_INCLUDE_DIR)

# Header Files
target_sources(soil PUBLIC FILE_SET public_headers TYPE HEADERS FILES
  soillib/soillib.hpp
  soillib/core/buffer.hpp
  soillib/core/index.hpp
  soillib/core/types.hpp
  soillib/index/flat.hpp
  soillib/index/quad.hpp
  soillib/index/layer.hpp
  soillib/io/geotiff.hpp
  soillib/io/mesh.hpp
  soillib/io/tiff.hpp
  soillib/io/yaml.hpp
  soillib/op/common.hpp
  soillib/op/erosion.hpp
  soillib/op/flow.hpp
  soillib/op/gather.hpp
  soillib/op/math.hpp
  soillib/op/noise.hpp
  soillib/op/normal.hpp
  soillib/util/error.hpp
  soillib/util/texture.hpp
  soillib/util/timer.hpp
  soillib/util/yield.hpp
  soillib/external/FastNoiseLite.h
)

# CUDA Files
target_sources(soil PUBLIC
  ${CMAKE_SOURCE_DIR}/source/soillib/core/buffer.cu
  ${CMAKE_SOURCE_DIR}/source/soillib/op/flow.cu
  ${CMAKE_SOURCE_DIR}/source/soillib/op/erosion.cu
  ${CMAKE_SOURCE_DIR}/source/soillib/op/math.cu
  ${CMAKE_SOURCE_DIR}/source/soillib/op/common.cu
)

target_include_directories(soil PRIVATE
  ${GLM_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}    # For cuda_runtime.h
)

install(
  TARGETS soil
  FILE_SET public_headers
)