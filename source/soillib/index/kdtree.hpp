#ifndef SOILLIB_INDEX_KDTREE
#define SOILLIB_INDEX_KDTREE

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

namespace soil {

struct payload_t {

  float2 p;
  size_t i;

};

//! kdtree is a wrapper for a kdtree type
//!
//! the type is built so that it works by returning indices,
//! which can be used to then index separate buffers.
struct kdtree {

  kdtree(const soil::buffer& buffer){
    this->data = buffer_t<payload_t>(buffer.elem(), soil::host_t::GPU);
    this->setup(buffer.as<vec2>());
  }

  //
  // Allocate / Deallocate
  //

  //! Construct the kdtree from buffer data
  //!
  //! Note that the data is unchanged, and any query operations
  //! return a buffer of indices which align to this buffer.
  void setup(const buffer_t<vec2>& buffer);

  //! Execute a k-nearest neighbors search, returning indices
  buffer_t<int> knn(const buffer& query, const size_t k) const {
    return this->knnq(query.as<vec2>(), k);
  }

  buffer_t<int> knnq(const buffer_t<vec2>& query, const size_t k) const;

  //
  // Data Inspection
  //

  size_t elem() const { return this->data.elem(); }

private:
  buffer_t<payload_t> data; //!< kdtree structure with index
};

}

#endif