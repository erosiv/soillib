#ifndef SOILLIB_INDEX_KDTREE
#define SOILLIB_INDEX_KDTREE

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

// #include <cukd/builder.h>
// #include <cukd/knn.h>

namespace soil {

struct payload_t {
  float3 p;
  size_t i;
};

//! kdtree is a wrapper for a kdtree type
//!
//! the type is built so that it works by returning indices,
//! which can be used to then index separate buffers.
struct kdtree {

  kdtree(const soil::buffer& buffer){
      this->allocate(buffer.elem());
      this->setup(buffer.as<vec3>());
  }

  size_t elem() const { return this->_elem; }

  //
  // Allocate / Deallocate
  //

  void allocate(const size_t elem); //!< Allocate Data
  void deallocate();                //!< Deallocate Data

  void setup(const buffer_t<vec3>& buffer);

  /*
  buffer knn(const buffer& query, const size_t k) const {
    
  //    if(query.type() != soil::VEC3)
  //      throw soil::error::mismatch_type(query.type(), soil::VEC3);
  
  const auto query_t = query.as<vec3>();
  return soil::buffer(knnquery(this->buffer, query_t, k));
  
  }
  */

  //  soil::buffer_t<vec3> knnquery(const soil::buffer_t<vec3>& data, const soil::buffer_t<vec3>& query, const size_t k);
  
private:
  size_t _elem = 0;
  payload_t* data = NULL;
};

/*
//
// KDTree Implementation
//

namespace {

__global__ void _knnquery(const soil::buffer_t<vec3> query_b, const soil::buffer_t<vec3> data, soil::buffer_t<vec3> output, const size_t K){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= query_b.elem()) return;

  const float3* data_ptr = (float3*)data.data();
  const size_t N = data.elem();
  const vec3 q = query_b[n];

  // Candidate List, Query Result
  cukd::HeapCandidateList<16> result(100.0);
  cukd::stackBased::knn(result, make_float3(q.x, q.y, q.z), data_ptr, N);

  for(int k = 0; k < K; ++k) {
    int ID = result.get_pointID(k);
    vec3 r = ID < 0
      ? vec3(0.f,0.f,0.f)
      : data[ID];
    output[n * K + k] = r;

  }

}

}

void knnbuild(soil::buffer_t<vec3>& buffer){
  cukd::buildTree((float3*)buffer.data(), buffer.elem());
}

soil::buffer_t<vec3> knnquery(const soil::buffer_t<vec3>& data, const soil::buffer_t<vec3>& query, const size_t k) {

  const size_t n_query = query.elem()/3;
  auto output = soil::buffer_t<vec3>{k * n_query, soil::host_t::GPU};
  _knnquery<<<block(n_query, 512), 512>>>(query, data, output, k);
  return output;

}

*/

}

#endif