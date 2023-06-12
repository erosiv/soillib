#ifndef SOILLIB_MAP_BASIC
#define SOILLIB_MAP_BASIC

#include <soillib/soillib.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/slice.hpp>

namespace soil {
namespace map {

struct basic_config {
  glm::ivec2 dimension = glm::ivec2(0);
  float scale = 0.0f;
};

template<typename T, soil::index_t Index> struct basic_iterator;
template<typename T, soil::index_t Index = soil::index::flat>
struct basic {

  typedef Index index;
  typedef basic_config config;

  const glm::ivec2 dimension;
  const float scale;

  const size_t area = dimension.x*dimension.y;

  soil::slice<T, Index> slice;

  basic(const glm::ivec2 dimension, const float scale):
    dimension(dimension),
    scale(scale)
  {}
  basic(const config config):
    basic(config.dimension, config.scale)
  {}

  inline T* get(const glm::ivec2 p) noexcept {
    return slice.get(p);
  }

  const inline bool oob(const glm::ivec2 p) noexcept {
    return slice.oob(p);
  }

  basic_iterator<T, Index> begin() const noexcept { return basic_iterator<T, Index>(slice.begin(), dimension); }
  basic_iterator<T, Index> end()   const noexcept { return basic_iterator<T, Index>(slice.end(), dimension); }

};

template<typename T, soil::index_t Index>
struct basic_iterator {

  const glm::ivec2 res;

  slice_iterator<T, Index> iter = NULL;
  int ind = 0;

  basic_iterator() noexcept : iter(NULL){};
  basic_iterator(const slice_iterator<T, Index>& iter, const glm::ivec2 res) noexcept : iter(iter), res(res){};

  // Base Operators

  const basic_iterator<T, Index>& operator++() noexcept {
    ++iter;
    ++ind;
    return *this;
  };

  const bool operator!=(const basic_iterator<T, Index> &other) const noexcept {
    return this->iter != other.iter;
  };

  const slice_t<T> operator*() noexcept {
      return *iter;
  };

};

// Configuration Loading

#ifdef SOILLIB_IO_YAML

bool operator<<(basic_config& conf, soil::io::yaml::node& node){
  try {
    conf.dimension.x = node["dimension"][0].As<int>();
    conf.dimension.y = node["dimension"][1].As<int>();
    conf.scale = node["scale"].As<float>();
  } catch(soil::io::yaml::exception& e){
    return false;
  }
  return true;
}

#endif

}; // namespace map
}; // namespace soil

#endif
