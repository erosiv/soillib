#ifndef SOILLIB_UTIL_BUF
#define SOILLIB_UTIL_BUF

#include <soillib/soillib.hpp>

namespace soil {

/************************************************
buf is an owning raw-data extent,
with a built-in iterator. This acts as a basis
for a nuber of memory-pooling concepts.
************************************************/

template<typename T> struct buf_iterator;
template<typename T> struct buf {

  T* start = NULL;
  size_t size = 0;

  buf(size_t size):size(size){
    start = new T[size];
  }

  ~buf(){
    delete[] start;
  }

  const buf_iterator<T> begin() const noexcept { return buf_iterator<T>(start, 0); }
  const buf_iterator<T> end()   const noexcept { return buf_iterator<T>(start, size); }

};

template<typename T> struct buf_iterator {

  T* start;
  size_t ind;

  buf_iterator() noexcept : start(NULL),ind(0){};
  buf_iterator(T* t, size_t ind) noexcept : start(t),ind(ind){};

  // Base Operators

  T& operator*() noexcept {
    return *(this->start+ind);
  };

  const buf_iterator<T>& operator++() noexcept {
    ind++;
    return *this;
  };

  const bool operator==(const buf_iterator<T>& other) const noexcept {
    return this->start == other.start && this->ind == other.ind;
  };

  const bool operator!=(const buf_iterator<T>& other) const noexcept {
    return !(*this == other);
  };

};

}; // end of namespace

#endif
