#ifndef SOILLIB_MAP_BASIC
#define SOILLIB_MAP_BASIC

#include <soillib/soillib.hpp>
#include <soillib/util/index.hpp>
#include <soillib/util/slice.hpp>
#include <soillib/util/pool.hpp>

namespace soil {
namespace map {

template<typename T, soil::index_t Index> struct basic_iterator;
template<typename T, soil::index_t Index = soil::index::flat>
struct basic {

  const size_t size = 1024;
  const size_t area = size*size;
  const glm::ivec2 dimension = glm::ivec2(size);
  soil::slice<T, Index> slice;

  basic(){}
  basic(soil::pool<T>& pool){
    slice = {pool.get(area), dimension};
  }

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

}; // namespace map
}; // namespace soil

#endif
