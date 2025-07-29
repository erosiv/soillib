#
# Python Binding Generation
#

#
# Load Dependencies
#

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
add_subdirectory(${CMAKE_SOURCE_DIR}/ext/nanobind nanobind)

#
# Define the Module and Bindings
#

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

# Internal Target Dependencies
find_package(glm CONFIG REQUIRED)

target_link_libraries(${MODULE_NAME} PRIVATE soil)
target_link_libraries(${MODULE_NAME} PRIVATE glm::glm)

# libtiff wrangling: there has to be a better way...
set(LIBTIFF_DIR "C:/Program Files (x86)/tiff")
find_package(tiff REQUIRED)
target_include_directories(${MODULE_NAME} PUBLIC
  "${LIBTIFF_DIR}/include"
)
target_link_libraries(${MODULE_NAME} PUBLIC 
  "${LIBTIFF_DIR}/lib/tiffd.lib"
)

# Install Directive for Scikit
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${MODULE_NAME})

# Install Runtime Dependencies into Wheel File
install(TARGETS nanobind RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/soillib)
install(TARGETS soil RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/soillib)
install(FILES "${LIBTIFF_DIR}/bin/tiffd.dll" DESTINATION ${SKBUILD_PLATLIB_DIR}/soillib)