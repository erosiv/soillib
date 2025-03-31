#
# soillib C++ Library Target
#

add_library(soil STATIC)
set_target_properties(soil PROPERTIES 
  LINKER_LANGUAGE CUDA
)
set(CMAKE_CUDA_FLAGS "-diag-suppress 20012 -diag-suppress 20013 -diag-suppress 20015 -diag-suppress 445 -diag-suppress 177 -fPIC -c -O3")

# Dependency Wrangling
find_path(GLM_INCLUDE_DIR glm/glm.hpp)
if(GLM_INCLUDE_DIR)
  message(STATUS "GLM Install Location: ${GLM_INCLUDE_DIR}")
  target_include_directories(soil PRIVATE ${GLM_INCLUDE_DIR})
else()
  message(FATAL_ERROR "glm is not installed. Please install glm.")
endif(GLM_INCLUDE_DIR)

target_link_libraries(soil cudaKDTree)

# Install Header Files
set(SOILLIB_DIR ${CMAKE_SOURCE_DIR}/source/soillib)
install(DIRECTORY ${SOILLIB_DIR} DESTINATION include)

# Build and Install Library
target_include_directories(soil PRIVATE
  ${CMAKE_SOURCE_DIR}/source
  ${CUDA_INCLUDE_DIRS}    # For cuda_runtime.h
)

target_sources(soil PUBLIC
  ${CMAKE_SOURCE_DIR}/source/soillib/core/buffer.cu
  ${CMAKE_SOURCE_DIR}/source/soillib/op/flow.cu
  ${CMAKE_SOURCE_DIR}/source/soillib/op/erosion.cu
  ${CMAKE_SOURCE_DIR}/source/soillib/op/math.cu
  ${CMAKE_SOURCE_DIR}/source/soillib/op/common.cu
  ${CMAKE_SOURCE_DIR}/source/soillib/op/pointcloud.cu
  ${CMAKE_SOURCE_DIR}/source/soillib/op/rbf.cu
)

install(
  TARGETS soil
)