#ifndef SOILLIB_OP_RBF_CU
#define SOILLIB_OP_RBF_CU
#define HAS_CUDA

#include <curand_kernel.h>
#include <soillib/op/rbf.hpp>
#include <soillib/op/common.hpp>

#include <cukd/builder.h>
#include <cukd/knn.h>

// Radial Basis Function Interpolator:
//  Supports Fitting of Data-Points with Non-Aligned Centroids,
//  as well as data sampling.
//
// Next Steps:
//  Implement gradient-descent for the centroid positions as well.
//  This should improve the quality of the fit tremendously.
//  Finally, implement the shape function gradient as well. 
//  As long as the number of centroids is less than the number of
//  data points, this convergence should be stable.

namespace soil {

//
// Initialize RBF Centroids
//

namespace {

__global__ void rbf_init(rbf rbf, const soil::buffer_t<vec2> center_b){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= center_b.elem()) return;
  rbf.weights[n] = 0.0f;
  rbf.centers[n] = center_b[n];
}

}

void rbf::init(const buffer_t<vec2>& centers){

  const size_t elem = centers.elem();
  this->elem = elem;

  this->weights = soil::buffer_t<float>(elem, soil::host_t::GPU);
  this->centers = soil::buffer_t<vec2>(elem, soil::host_t::GPU);

  rbf_init<<<block(elem, 1024), 1024>>>(*this, centers);

}

//
// Matrix Computation
//

namespace {

__global__ void rbf_matrix(rbf rbf, soil::buffer_t<float> matrix_b, const soil::buffer_t<vec2> samples_b, const soil::flat_t<2> index, const size_t K, const size_t N) {

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= N*K) return;
  
  soil::ivec2 entry = index.unflatten(i);
  const size_t n = entry[0];
  const size_t k = entry[1];

  const vec2 pos = samples_b[n];
  const vec2 c = rbf.centers[k];
  const float s = rbf.shape;
  const float r = glm::length(c - pos);
  matrix_b[i] = rbf::func(r / s);

}

}

buffer_t<float> rbf::matrix(const buffer_t<vec2>& samples) const {

  const size_t K = this->elem;
  const size_t N = samples.elem();
  buffer_t<float> matrix = buffer_t<float>{ N*K, soil::host_t::GPU };

  const auto index_t = soil::flat_t<2>(vec2(N, K));
  rbf_matrix<<<block(N*K, 1024), 1024>>>(*this, matrix, samples, index_t, K, N);

  return matrix;

}

//
// RBF Fitting Procedure
//

namespace {

struct payload_traits: 
  public cukd::default_data_traits<kdtree::pnt_t> {
 
  using point_t = kdtree::pnt_t;

  static inline __device__ __host__
  point_t get_point(const payload_t &data)
  { return data.p; }

  static inline __device__ __host__
  float get_coord(const payload_t &data, int dim)
  { return cukd::get_coord(get_point(data),dim); }

  enum { has_explicit_dim = false };
  static inline __device__ int  get_dim(const payload_t &) { return -1; }

};

template<size_t K>
__device__ void knn(const soil::kdtree& kdtree, const vec2 pos, cukd::HeapCandidateList<K>& list) {

  cukd::stackBased::knn<
    cukd::HeapCandidateList<K>,
    payload_t,
    payload_traits
  >(list, make_point(pos), kdtree.data.data(), kdtree.elem());

}

__global__ void zero_delta(soil::buffer_t<float> delta_b){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= delta_b.elem()) return;
  delta_b[n] = 0.0f;
}

__device__ float rbf_sample(const soil::kdtree& kdtree, const rbf& rbf, const vec2& pos){

  // Closest Point Accumulation:

  /*
  const size_t B = 64;
  const float rad = rbf.shape * 10.0f;
  cukd::HeapCandidateList<B> list(rad);
  knn<B>(kdtree, pos, list);
  
  float val = 0.0f;
  for(int b = 0; b < B; ++b) {
    
  int k = list.get_pointID(b);
  if(k >= 0){
    k = kdtree.data[k].i;
    
    // accumulate into val!
    const vec2 c = rbf.centers[k];
    const float w = rbf.weights[k];
    const float s = rbf.shape;
    const float r = glm::length(c - pos);
    val += w * rbf::func(r / s);
    
  }
  
}

return val;
*/

  // Full Accumulation:
  const size_t K = rbf.elem;
  
  float val = 0.0f;
  for(int k = 0; k < K; ++k) {
    
    // accumulate into val!
    const vec2 c = rbf.centers[k];
    const float w = rbf.weights[k];
    const float s = rbf.shape;
    const float r = glm::length(c - pos);
    val += w * rbf::func(r / s);
  
  }

  return val;

}

}

//
// Radial Basis Function Sampling
//

namespace {

__global__ void rbf_sample(const soil::kdtree kdtree, const rbf rbf, const soil::buffer_t<vec2> pos_b, soil::buffer_t<float> val_b){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= pos_b.elem()) return;

  const size_t N = pos_b.elem();
  const size_t K = kdtree.elem();

  const vec2 pos = pos_b[n];
  val_b[n] = rbf_sample(kdtree, rbf, pos);

}

__global__ void rbf_sample(const soil::kdtree kdtree, const rbf rbf, const soil::flat_t<2> index, soil::buffer_t<float> val_b){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= index.elem()) return;

  const size_t N = index.elem();
  const size_t K = kdtree.elem();

  const vec2 pos = index.unflatten(n);
  val_b[n] = rbf_sample(kdtree, rbf, pos);

}

}
  
buffer_t<float> soil::rbf::sample(const soil::kdtree& kdtree, const buffer_t<vec2>& pos) const {

  const size_t N = pos.elem();
  auto values = soil::buffer_t<float>(N, soil::host_t::GPU);
  rbf_sample<<<block(N, 1024), 1024>>>(kdtree, *this, pos, values);
  return values;

}

buffer_t<float> soil::rbf::sample(const soil::kdtree& kdtree, const soil::flat_t<2>& index) const {

  const size_t N = index.elem();
  auto values = soil::buffer_t<float>(N, soil::host_t::GPU);
  rbf_sample<<<block(N, 1024), 1024>>>(kdtree, *this, index, values);
  return values;

}

}

#endif