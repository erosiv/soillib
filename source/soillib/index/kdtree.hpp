#ifndef SOILLIB_INDEX_KDTREE
#define SOILLIB_INDEX_KDTREE

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

namespace soil {

template<typename T>
concept vectortype = (
//  std::same_as<T, soil::vec3> ||
  std::same_as<T, soil::vec2>
);

namespace {
#ifdef HAS_CUDA

__device__ float2 make_point(const vec2 vec){
  return make_float2(vec.x, vec.y);
}

__device__ float3 make_point(const vec3 vec){
  return make_float3(vec.x, vec.y, vec.z);
}

#endif
}

struct payload_t {
  float2 p;
  size_t i;
};

//! kdtree is a wrapper for a kdtree type
//!
//! the type is built so that it works by returning indices,
//! which can be used to then index separate buffers.
struct kdtree {

  using pnt_t = float2;
  using vec_t = soil::vec2;
  static constexpr int vec_n = 2;

  kdtree(const soil::buffer& buffer) {
    soil::select(buffer.type(), [&]<vectortype T>(){
      this->data = buffer_t<payload_t>(buffer.elem(), soil::host_t::GPU);
      this->setup(buffer.as<vec_t>());
    });
  }

  //! Construct the kdtree from buffer data
  //!
  //! Note that the data is unchanged, and any query operations
  //! return a buffer of indices which align to this buffer.
  void setup(const buffer_t<vec_t>& buffer);

  //! Execute a k-nearest neighbors search, returning indices
  buffer_t<int> knn(const buffer& query, const size_t k) const {
    const size_t n_query = query.elem()/vec_n;
    auto indices = soil::buffer_t<int>{k * n_query, soil::host_t::GPU};
    this->knni(query.as<vec_t>(), k, indices);
    return indices;
  }

  //! Execute a k-nearest neighbors with existing indices
  void knni(const buffer_t<vec_t>& query, const size_t k, buffer_t<int>& indices) const;

  //
  // Data Inspection
  //

  GPU_ENABLE size_t elem() const { return this->data.elem(); }

//private:
  buffer_t<payload_t> data; //!< kdtree structure with index
};

}

#endif