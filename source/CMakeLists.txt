#
# soillib C++ Library Target
#

add_library(soil STATIC)
set_target_properties(soil PROPERTIES 
  LINKER_LANGUAGE CUDA
)

target_include_directories(soil PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/soillib>
  .
)

target_sources(soil PUBLIC
  soillib/op/erosion.cu
  soillib/op/math.cu
  soillib/op/common.cu
# soillib/op/flow.cu
)

# Dependency Wrangling

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# target_compile_options(soil PUBLIC -diag-suppress 20013)
# target_compile_options(soil PUBLIC --expt-relaxed-constexpr)

# Build and Install Library
# target_include_directories(soil PUBLIC
#   ${CUDA_INCLUDE_DIRS}    # For cuda_runtime.h
# )

find_package(glm CONFIG REQUIRED)
target_link_libraries(soil PRIVATE glm::glm)