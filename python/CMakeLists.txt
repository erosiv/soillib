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
  source/buffer.cpp
  source/index.cpp
  source/io.cpp
  source/util.cpp
)

target_compile_definitions(${MODULE_NAME} PRIVATE
MODULE_NAME=${MODULE_NAME})

# Internal Target Dependencies
find_package(glm CONFIG REQUIRED)

target_link_libraries(${MODULE_NAME} PRIVATE soil)
target_link_libraries(${MODULE_NAME} PRIVATE glm::glm)

# libtiff wrangling: there has to be a better way...
find_package(tiff REQUIRED)
target_include_directories(${MODULE_NAME} PUBLIC
  "C:\\Program Files (x86)\\tiff\\include"
)
target_link_libraries(${MODULE_NAME} PUBLIC 
  "C:\\Program Files (x86)\\tiff\\lib\\tiffd.lib"
)

add_custom_command (
  TARGET ${MODULE_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:${MODULE_NAME}>"
  "C:\\Program Files (x86)\\glew-2.1.0\\bin\\Release\\x64\\glew32.dll"
)


# Install Directive for Scikit
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${MODULE_NAME})