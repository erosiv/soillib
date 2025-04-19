#ifndef SOILLIB_INDEX_KDTREE_CU
#define SOILLIB_INDEX_KDTREE_CU
#define HAS_CUDA

#include <soillib/index/kdtree.hpp>
#include <cukd/builder.h>
#include <cukd/knn.h>

namespace soil {

void kdtree::deallocate(){
  if(this->data != NULL){
    cudaFree(this->data);
    this->data = NULL;
    this->_elem = 0;
  }
}

void kdtree::allocate(const size_t elem){
  cudaMalloc(&this->data, elem * sizeof(payload_t));
  this->_elem = elem;
}

//
// KDTree Implementation
//

struct PointPlusPayload_traits
: public cukd::default_data_traits<float3>
{
using point_t = float3;

static inline __device__ __host__
float3 get_point(const payload_t &data)
{ return data.p; }

static inline __device__ __host__
float  get_coord(const payload_t &data, int dim)
{ return cukd::get_coord(get_point(data),dim); }

enum { has_explicit_dim = false };

/*! !{ just defining this for completeness, get/set_dim should never
  get called for this type because we have set has_explicit_dim
  set to false. note traversal should ONLY ever call this
  function for data_t's that define has_explicit_dim to true */
static inline __device__ int  get_dim(const payload_t &) { return -1; }
};

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

__global__ void _knncopy(payload_t* out, const size_t N, const soil::buffer_t<vec3> source) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;
  const vec3 p = source[n];
  out[n].p = make_float3(p.x, p.y, p.z);
  out[n].i = n;
}

}

void kdtree::setup(const buffer_t<vec3>& source) {

  std::cout<<"Copying Data..."<<std::endl;
  _knncopy<<<block(this->elem(), 1024), 1024>>>(this->data, this->elem(), source);

  std::cout<<"Building Tree..."<<std::endl;
  cukd::buildTree<
    payload_t,
    PointPlusPayload_traits
  >(this->data, this->elem());

}

namespace {

__global__ void _knnquery(const soil::buffer_t<vec3> query_b, const payload_t* data, const size_t N, soil::buffer_t<vec3> output, const size_t K){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= query_b.elem()) return;

  const payload_t* data_ptr = data;
  const vec3 q = query_b[n];

  // Candidate List, Query Result
  cukd::HeapCandidateList<16> result(100.0);
  cukd::stackBased::knn<
    cukd::HeapCandidateList<16>,
    payload_t,
    PointPlusPayload_traits
  >(result, make_float3(q.x, q.y, q.z), data_ptr, N);

  for(int k = 0; k < K; ++k) {
    int ID = result.get_pointID(k);
    if(ID < 0){
      output[n * K + k] = vec3(0.0f);
    } else {
      float3 r = data[ID].p;
      output[n * K + k] = vec3(r.x, r.y, r.z);
    }
  }

}

}

buffer_t<vec3> kdtree::knnq(const buffer_t<vec3>& query, const size_t k) const {

  const size_t n_query = query.elem()/3;
  auto output = soil::buffer_t<vec3>{k * n_query, soil::host_t::GPU};
  _knnquery<<<block(n_query, 512), 512>>>(query, this->data, this->elem(), output, k);
  return output;

}

}

#endif