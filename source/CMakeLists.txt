#
# soillib C++ Library Target
#

set(TARGET_NAME soil)
add_library(${TARGET_NAME} STATIC)
set_target_properties(${TARGET_NAME} PROPERTIES 
  LINKER_LANGUAGE CUDA
)

# target_compile_definitions(${TARGET_NAME} PRIVATE -DSHARED_BUILD)
target_include_directories(${TARGET_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/soillib>
  .
)

target_sources(${TARGET_NAME} PUBLIC
  soillib/model/erosion.cu
  soillib/model/graph/graph.cu
  soillib/model/filter/filter.cu
# soillib/op/flow.cu
)

find_package(glm CONFIG REQUIRED)
target_link_libraries(soil PRIVATE glm::glm)

add_subdirectory(${CMAKE_SOURCE_DIR}/ext/silt silt)
target_link_libraries(${TARGET_NAME} PUBLIC silt_lib)