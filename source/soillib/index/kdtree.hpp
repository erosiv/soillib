#ifndef SOILLIB_INDEX_KDTREE
#define SOILLIB_INDEX_KDTREE

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

namespace soil {

template<typename T>
concept vectortype = (
//  std::same_as<T, soil::vec2> ||
  std::same_as<T, soil::vec3>
);

struct payload_t {
  float3 p;
  size_t i;
};

//! kdtree is a wrapper for a kdtree type
//!
//! the type is built so that it works by returning indices,
//! which can be used to then index separate buffers.
struct kdtree {

  kdtree(const soil::buffer& buffer) {
    soil::select(buffer.type(), [&]<vectortype T>(){
      this->data = buffer_t<payload_t>(buffer.elem(), soil::host_t::GPU);
      this->setup(buffer.as<vec3>());
    });
  }

  //
  // Allocate / Deallocate
  //

  //! Construct the kdtree from buffer data
  //!
  //! Note that the data is unchanged, and any query operations
  //! return a buffer of indices which align to this buffer.
  void setup(const buffer_t<vec3>& buffer);

  //! Execute a k-nearest neighbors search, returning indices
  buffer_t<int> knn(const buffer& query, const size_t k) const {
    const size_t n_query = query.elem()/3;
    auto indices = soil::buffer_t<int>{k * n_query, soil::host_t::GPU};
    this->knni(query.as<vec3>(), k, indices);
    return indices;
  }

  //! Execute a k-nearest neighbors with existing indices
  void knni(const buffer_t<vec3>& query, const size_t k, buffer_t<int>& indices) const;

  //
  // Data Inspection
  //

  size_t elem() const { return this->data.elem(); }

private:
  buffer_t<payload_t> data; //!< kdtree structure with index
};

}

#endif