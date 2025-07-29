#
# soillib C++ Library Target
#

add_library(soil SHARED)
set_target_properties(soil PROPERTIES 
  LINKER_LANGUAGE CUDA
)
target_compile_definitions(soil PRIVATE -DSHARED_BUILD)
target_include_directories(soil PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/soillib>
  .
)

target_sources(soil PUBLIC
#  soillib/model/erosion.cu
  soillib/op/common.cu
# soillib/op/flow.cu
)

find_package(glm CONFIG REQUIRED)
target_link_libraries(soil PRIVATE glm::glm)