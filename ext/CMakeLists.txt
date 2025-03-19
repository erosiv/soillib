#
# nanobind C++ Library
#

add_library(nanobind STATIC)
set_target_properties(nanobind PROPERTIES 
  LINKER_LANGUAGE CXX
  CMAKE_CXX_STANDARD 17
)
set(NANOBIND_DIR ${CMAKE_SOURCE_DIR}/ext/nanobind)

target_include_directories(nanobind PUBLIC
  ${NANOBIND_DIR}/ext/robin_map/include
  ${NANOBIND_DIR}/include
)

set(CMAKE_CXX_FLAGS "-fvisibility=hidden \
    	-DNDEBUG -DNB_COMPACT_ASSERTIONS \
    	`python3-config --includes` -fPIC \
			-O3 -fno-strict-aliasing \
      -ffunction-sections -fdata-sections
")

target_sources(nanobind PUBLIC
  ${NANOBIND_DIR}/src/nb_combined.cpp
)

install(
  TARGETS nanobind
#  FILE_SET public_headers
)