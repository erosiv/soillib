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
  NB_STATIC STABLE_ABI
  source/_module.cpp 
  source/buffer.cpp
  source/index.cpp
  source/io.cpp
  source/op.cpp
  source/util.cpp
)

set_property(TARGET ${MODULE_NAME} APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:nanobind>")
set_property(TARGET ${MODULE_NAME} APPEND PROPERTY INSTALL_RPATH ".. ?? ..")

target_compile_definitions(${MODULE_NAME} PRIVATE
  MODULE_NAME=${MODULE_NAME})

# Internal Target Dependencies
target_link_libraries(${MODULE_NAME} PRIVATE soil)

find_package(glm CONFIG REQUIRED)
target_link_libraries(${MODULE_NAME} PRIVATE glm::glm)

find_package(tiff REQUIRED)
target_include_directories(${MODULE_NAME} PRIVATE
  "C:\\Program Files (x86)\\tiff\\include"
)
# target_link_libraries(${MODULE_NAME} PUBLIC tiffd)
target_link_libraries(${MODULE_NAME} PRIVATE "C:\\Program Files (x86)\\tiff\\lib\\tiffd.lib")

# Install Directive for Scikit
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${MODULE_NAME})