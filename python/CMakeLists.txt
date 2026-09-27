#
# Python Binding Generation
#

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#
# Dependencies
#

# 3.12 is the floor because the wheel is built as abi3/cp312
# (wheel.py-api in pyproject.toml). Keep these two in step.
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT TARGET nanobind::nanobind_static AND NOT TARGET nanobind)
  add_subdirectory(${CMAKE_SOURCE_DIR}/ext/nanobind nanobind)
endif()

#
# Module
#

set(MODULE_NAME soillib)
nanobind_add_module(${MODULE_NAME}
  NB_SHARED
  NB_DOMAIN "erosiv"
  source/_module.cpp
  source/io.cpp
  source/model.cpp
  source/util.cpp
)

# Linux RPATH configuration
set_target_properties(${MODULE_NAME} PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH FALSE
)

target_compile_definitions(${MODULE_NAME} PRIVATE
  MODULE_NAME=${MODULE_NAME})

# glm and silt arrive transitively through soillib::soillib.
target_link_libraries(${MODULE_NAME} PRIVATE soillib::soillib)

#
# libtiff
#

set(_soil_saved_build_shared "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS ON)
if(NOT TARGET TIFF::tiff)
  add_subdirectory(${CMAKE_SOURCE_DIR}/ext/libtiff libtiff)
endif()
set(BUILD_SHARED_LIBS "${_soil_saved_build_shared}")
unset(_soil_saved_build_shared)

target_link_libraries(${MODULE_NAME} PRIVATE TIFF::tiff)

#
# Co-locate dependent DLLs with the module in the BUILD TREE (Windows).
#

if(WIN32)
add_custom_command(TARGET ${MODULE_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_RUNTIME_DLLS:${MODULE_NAME}>"
    "$<TARGET_FILE_DIR:${MODULE_NAME}>"
  COMMAND_EXPAND_LISTS
)
endif()

#
# Type Stubs (.pyi)
#

nanobind_add_stub(
  ${MODULE_NAME}_stub
  MODULE ${MODULE_NAME}
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_NAME}/${MODULE_NAME}.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:${MODULE_NAME}>
  DEPENDS ${MODULE_NAME}
)

#
# Installation (scikit-build-core)
#

install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${MODULE_NAME})

# libtiff is soillib's own dependency and ships in soillib's wheel.
install(TARGETS tiff
  RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/${MODULE_NAME}
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/${MODULE_NAME}
)