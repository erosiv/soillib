#ifndef SOILLIB_MAP_BASIC
#define SOILLIB_MAP_BASIC

#include <soillib/soillib.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/slice.hpp>
#include <soillib/util/pool.hpp>

namespace soil {
namespace map {

template<typename T, soil::index_t Index>
struct basic {

  const glm::ivec2 size = 1024;
  const glm::ivec2 area = size*size;
  const glm::ivec2 dimension = glm::ivec2(size);
  soil::slice<T, Index> slice;

  basic(){}
  basic(soil::pool<T>& pool){
    slice = {pool.get(area), dimension};
  }

  inline T* get(const glm::ivec2 p) const noexcept {
    return slice.get(p);
  }

  const inline bool oob(const glm::ivec2 p) noexcept {
    return slice.oob(p);
  }

};

}; // namespace map
}; // namespace soil

#endif
