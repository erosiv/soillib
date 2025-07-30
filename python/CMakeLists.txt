#
# Python Binding Generation
#

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#
# Define the Module and Bindings
#

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory(${CMAKE_SOURCE_DIR}/ext/nanobind nanobind)

set(MODULE_NAME soillib)
nanobind_add_module(${MODULE_NAME}
  source/_module.cpp
  source/shape.cpp
  source/tensor.cpp
  source/io.cpp
  source/op.cpp
#  source/model.cpp
  source/util.cpp
  NB_SHARED
)
target_compile_definitions(${MODULE_NAME} PRIVATE
  MODULE_NAME=${MODULE_NAME})

#
# Dependencies
#

target_link_libraries(${MODULE_NAME} PRIVATE soil)

add_subdirectory(${CMAKE_SOURCE_DIR}/ext/libtiff libtiff)
find_package(tiff CONFIG REQUIRED)
target_link_libraries(${MODULE_NAME} PRIVATE tiff)

find_package(glm CONFIG REQUIRED)
target_link_libraries(${MODULE_NAME} PRIVATE glm::glm)

#
# Installation
#

# Install Directive for Scikit
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${MODULE_NAME})

# Install Runtime Dependencies into Wheel File
install(TARGETS nanobind RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/soillib)
install(TARGETS soil RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/soillib)
# install(TARGETS tiff RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/soillib)