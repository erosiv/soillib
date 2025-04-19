#
# nanobind C++ Library
#

add_subdirectory(cudaKDTree)

add_library(nanobind STATIC)
set_target_properties(nanobind PROPERTIES 
  LINKER_LANGUAGE CXX
  CMAKE_CXX_STANDARD 17
)

set(CMAKE_CXX_FLAGS "-fvisibility=hidden \
    	-DNDEBUG -DNB_COMPACT_ASSERTIONS \
    	`python3-config --includes` -fPIC \
			-O3 -fno-strict-aliasing \
      -ffunction-sections -fdata-sections
")

# Install Headers
set(NANOBIND_DIR ${CMAKE_SOURCE_DIR}/ext/nanobind)
install(
  DIRECTORY ${NANOBIND_DIR}/include/nanobind 
  DESTINATION include
)

# Build and Intall Library
target_include_directories(nanobind PUBLIC
  ${NANOBIND_DIR}/ext/robin_map/include
  ${NANOBIND_DIR}/include
)

target_sources(nanobind PUBLIC
  ${NANOBIND_DIR}/src/nb_combined.cpp
)

install(
  TARGETS nanobind
)