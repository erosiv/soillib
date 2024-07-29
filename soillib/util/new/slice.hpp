#ifndef SOILLIB_UTIL_SLICE
#define SOILLIB_UTIL_SLICE

#include <soillib/soillib.hpp>
#include <soillib/util/buf.hpp>
#include <soillib/util/index.hpp>

namespace soil {

/*
Now basically what happens is slice is a wrapper around buf,
that generates an interator based on the index number. Indices
are the actual structures which have to be multi-dimension right?

The idea is that for slice, we can define some generic "extent",
and a generic index which then generates an iterator for us.

That way an image is just a slice<2D> + Buf,
and we can generate everything dynamically if we want to.
*/

/*
How do we implement slices on top of these generic buffers?
I suppose that each 
*/

template<typename T>
struct slice {

  slice(buf){
  //   ... store the slice
  }

  // generate an iterator of a given shape
  // this is basically an iterable memory view

  // 
  inline size_t elem(){
    return /// ... the total number of elements of a type
  }

};

/*
buffer* buf = buffer::make("float", 64);
slice<float, glm::ivec2> s(buf, {1, 2});

for(auto& [value, pos]: slice<float, glm::ivec2>(buf, {1, 2})){

}
*/















/*
//! slice<T, I> is an n-dimensional, iterable data-view
//! which combines a buffer with an indexing procedure.
//!
//! It implements index-based bound- checking, safe and
//! un-safe access operators and forward-iteration w.
//! structured bindings to retrieve position and value.
//!
template<typename T, soil::index_t Index> struct slice_iterator;
template<typename T, soil::index_t Index>
struct slice {

  soil::buf<T> root;
  glm::ivec2 res = glm::ivec2(0);

  slice(const glm::ivec2 res)
    :res(res),root(res.x*res.y){}

  const inline size_t size() const {
    return res.x * res.y;
  }

  const inline bool oob(const glm::ivec2 p) const {
    return Index::oob(p, res);
  }

  inline T* get(const glm::ivec2 p) const {
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

*/


}; // end of namespace

#endif
