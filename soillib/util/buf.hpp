#ifndef SOILLIB_UTIL_BUF
#define SOILLIB_UTIL_BUF

#include <soillib/soillib.hpp>

namespace soil {

/************************************************
buf is an (optionally) owning raw-data extent,
with a built-in iterator. This acts as a basis
for a nuber of memory-pooling concepts.
************************************************/

template<typename T> struct buf_iterator;
template<typename T> struct buf {
  T* start = NULL;
  size_t size = 0;

  const buf_iterator<T> begin() const noexcept { return buf_iterator<T>(start); }
  const buf_iterator<T> end()   const noexcept { return buf_iterator<T>(start+size); }
};

template<typename T> struct buf_iterator {

  T* iter = NULL;

  buf_iterator() noexcept : iter(NULL){};
  buf_iterator(T* t) noexcept : iter(t){};

  const T operator*() noexcept {
      return *this->iter;
  };

  const buf_iterator<T>& operator++() noexcept {
    if(iter != NULL) ++iter;
    return *this;
  };

  const bool operator!=(const buf_iterator<T>& other) const noexcept {
    return this->iter != other.iter;
  };
};

}; // end of namespace

#endif
