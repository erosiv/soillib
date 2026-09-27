#
# soillib C++/CUDA Library Target
#

set(TARGET_NAME soil)
add_library(${TARGET_NAME} STATIC)
add_library(soillib::soillib ALIAS ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES
  LINKER_LANGUAGE CUDA
  POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(${TARGET_NAME} PUBLIC HAS_CUDA)

math(EXPR SOIL_VERSION_NUM
  "${soillib_VERSION_MAJOR} * 10000 + ${soillib_VERSION_MINOR} * 100 + ${soillib_VERSION_PATCH}")
target_compile_definitions(${TARGET_NAME} PUBLIC
  SOIL_VERSION_MAJOR=${soillib_VERSION_MAJOR}
  SOIL_VERSION_MINOR=${soillib_VERSION_MINOR}
  SOIL_VERSION_PATCH=${soillib_VERSION_PATCH}
  SOIL_VERSION_NUM=${SOIL_VERSION_NUM}
  SOIL_VERSION_STRING="${soillib_VERSION}"
)

target_include_directories(${TARGET_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/soillib>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ext/FastNoiseLite/Cpp>
)

target_sources(${TARGET_NAME} PRIVATE
  soillib/model/graph/graph.cu
  soillib/model/filter/filter.cu
  soillib/model/grad/grad.cu
  soillib/model/path/erosion.cu
  soillib/model/path/path.cu
  soillib/op/normal.cu
  soillib/op/resize.cu
)

#
# Dependencies
#

if(NOT TARGET silt::silt)
  add_subdirectory(${CMAKE_SOURCE_DIR}/ext/silt silt)
endif()

target_link_libraries(${TARGET_NAME} PUBLIC silt::silt)

find_package(CUDAToolkit)
target_link_libraries(${TARGET_NAME} PUBLIC CUDA::curand)