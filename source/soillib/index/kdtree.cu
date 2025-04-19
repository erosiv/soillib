#ifndef SOILLIB_INDEX_KDTREE_CU
#define SOILLIB_INDEX_KDTREE_CU
#define HAS_CUDA

#include <soillib/index/kdtree.hpp>

#include <cukd/builder.h>
#include <cukd/knn.h>

namespace soil {

//
// KDTree Implementation
//

struct PointPlusPayload_traits: public cukd::default_data_traits<float2> {
 
  using point_t = float2;

  static inline __device__ __host__
  point_t get_point(const payload_t &data)
  { return data.p; }

  static inline __device__ __host__
  float  get_coord(const payload_t &data, int dim)
  { return cukd::get_coord(get_point(data),dim); }

  enum { has_explicit_dim = false };
  static inline __device__ int  get_dim(const payload_t &) { return -1; }

};

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

__global__ void _kdtreecopy(buffer_t<payload_t> out, const soil::buffer_t<vec2> source) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= out.elem()) return;
  const vec2 p = source[n];
  out[n].p = make_float2(p.x, p.y);
  out[n].i = n;
}

__global__ void _knnquery(const buffer_t<vec2> query_b, const buffer_t<payload_t> data, soil::buffer_t<int> output, const size_t K){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= query_b.elem()) return;

  const vec2 q = query_b[n];

  // Candidate List, Query Result
  cukd::HeapCandidateList<16> result(100.0);
  cukd::stackBased::knn<
    cukd::HeapCandidateList<16>,
    payload_t,
    PointPlusPayload_traits
  >(result, make_float2(q.x, q.y), data.data(), data.elem());

  for(int k = 0; k < K; ++k) {
    int ID = result.get_pointID(k);
    if(ID < 0){
      output[n * K + k] = -1;
    } else {
      output[n * K + k] = data[ID].i;
    }
  }

}

}

void kdtree::setup(const buffer_t<vec2>& source) {
  _kdtreecopy<<<block(this->elem(), 1024), 1024>>>(this->data, source);
  cukd::buildTree<
    payload_t,
    PointPlusPayload_traits
  >(this->data.data(), this->elem());
}

buffer_t<int> kdtree::knnq(const buffer_t<vec2>& query, const size_t k) const {

  const size_t n_query = query.elem()/2;

  auto output = soil::buffer_t<int>{k * n_query, soil::host_t::GPU};
  _knnquery<<<block(n_query, 512), 512>>>(query, this->data, output, k);
  return output;

}

} // end of namespace soil::

#endif