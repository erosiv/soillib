#ifndef SOILLIB_MAP_BASIC
#define SOILLIB_MAP_BASIC

#include <soillib/soillib.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/slice.hpp>

/*==================================================
soillib map basic

simple rectangular map of arbitrary size, which uses
a cellpool to allocate templated cell data. Includes
a convenient iterator for accessing cell and pos.

the basic map is also templated by and index, which
lets you decide how the data chunk is ordered. This
also affects the iterator order.

it is yaml configurable by specifying the x and y
dimensions of the map.
==================================================*/

namespace soil {
namespace map {

// Configuration / Parameterization

struct basic_config {
  glm::ivec2 dimension = glm::ivec2(0);
};

// Basic Map

template<typename T, soil::index_t Index = soil::index::flat>
struct basic {

  typedef Index index;
  typedef basic_config config;

  const glm::ivec2 dimension;
  const size_t area = dimension.x*dimension.y;

  soil::slice<T, Index> slice;

  basic(const glm::ivec2 dimension)
    :dimension(dimension),slice(dimension){}

  basic(const config config)
    :basic(config.dimension){}

  inline T* get(const glm::ivec2 p) noexcept {
    return slice.get(p);
  }

  const inline bool oob(const glm::ivec2 p) noexcept {
    return slice.oob(p);
  }

  const inline glm::ivec2 bound() noexcept {
    return dimension;
  }

  slice_iterator<T, Index> begin() const noexcept { return slice.begin(); }
  slice_iterator<T, Index> end()   const noexcept { return slice.end(); }

};

}; // namespace map
}; // namespace soil

// Configuration Loading

#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<soil::map::basic_config> {
  static soil::map::basic_config As(soil::io::yaml& node){
    soil::map::basic_config config;
    config.dimension = node["dimension"].As<glm::ivec2>();
    return config;
  }
};

#endif
#endif
