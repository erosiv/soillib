#ifndef SOILLIB_UTIL_SLICE
#define SOILLIB_UTIL_SLICE

#include <soillib/soillib.hpp>
#include <soillib/util/buf.hpp>
#include <soillib/util/index.hpp>

namespace soil {

/************************************************
slice is an nD, iterable data-view which utilizes
an actual indexing methodology. Therefore, get
and put operations are based on positions in space.
************************************************/

template<typename T, soil::index_t Index> struct slice_iterator;
template<typename T, soil::index_t Index>
struct slice {

  soil::buf<T> root;
  glm::ivec2 res = glm::ivec2(0);

  const inline size_t size(){
    return res.x * res.y;
  }

  const inline bool oob(const glm::ivec2 p){
    return Index::oob(p, res);
  }

  inline T* get(const glm::ivec2 p){
    if(root.start == NULL) return NULL;
    if(Index::oob(p, res)) return NULL;
    return root.start + Index::flatten(p, res);
  }

  slice_iterator<T, Index> begin() const noexcept { return slice_iterator<T, Index>(root.begin(), res); }
  slice_iterator<T, Index> end()   const noexcept { return slice_iterator<T, Index>(root.end(), res); }

};

// Iterator and Iterator Reference Value

template<typename T> struct slice_t {
  T& start;
  const glm::ivec2 pos = glm::ivec2(0);
};

template<typename T, soil::index_t Index>
struct slice_iterator {

  buf_iterator<T> iter;  
  glm::ivec2 res;

  slice_iterator() noexcept : iter(NULL, 0){};
  slice_iterator(const buf_iterator<T>& iter, const glm::ivec2 res) noexcept : iter(iter),res(res){};

  // Base Operators

  const slice_t<T> operator*() noexcept {
    return {*iter, Index::unflatten(iter.ind, res)};
  };

  const slice_iterator<T, Index>& operator++() noexcept {
    ++iter;
    return *this;
  };

  const bool operator==(const slice_iterator<T, Index>& other) const noexcept {
    return this->iter == other.iter;
  };

  const bool operator!=(const slice_iterator<T, Index> &other) const noexcept {
    return !(*this == other);
  };

};

}; // end of namespace

#endif
