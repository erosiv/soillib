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

/*
Effectively, we have to use the kdtree to get the centroids
that are closest to the sample point, then use that to compute
the approximate value, get the error and compute the delta to
the weight function for each quantity nearest to us.
*/

//! Compute Delta Kernel
//!
//! For every data-point, this kernel finds the nearest points
//! and uses them to compute the local value. This gives the
//! quantified error, which is then added back to the various
//! nodes that are closest.
//!
__global__ void rbf_delta_w(const soil::kdtree kdtree, const rbf rbf, const soil::buffer_t<vec3> data_b, soil::buffer_t<float> delta_b){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= data_b.elem()) return;

  // Radial Basis Function Support Data

  const size_t N = data_b.elem();
  const size_t K = kdtree.elem();

  const vec3 data = data_b[n];
  const vec2 pos(data);

  // Sample Closest Points

  const size_t B = 16;
  cukd::HeapCandidateList<B> list(100.0);
  knn<B>(kdtree, pos, list);
  float val = 0.0f;

  // Closest Point Accumulation:
  for(int b = 0; b < B; ++b) {
    
    int k = list.get_pointID(b);
    if(k >= 0){
      k = kdtree.data[k].i;

      // accumulate into val!
      const vec2 c = rbf.centers[k];
      const float w = rbf.weights[k];
      const float s = rbf.shape;
      const float r = glm::length(c - pos);
      val += rbf::func(w, r, s);

    }

  }

  // this is the local error...
  const float err = val - data.z;

  // Closest Point Accumulation:
  for(int b = 0; b < B; ++b) {
  
    int k = list.get_pointID(b);
    if(k >= 0){

      const vec2 c = rbf.centers[k];
      const float w = rbf.weights[k];
      const float s = rbf.shape;
      const float r = glm::length(c - vec2(data_b[n]));
      const float grad = rbf::grad_w(w, r, s);
      atomicAdd(&delta_b[k], err * grad);

    }
  
  }

}

//! Descend Kernel
template<typename T>
__global__ void rbf_descend(buffer_t<T> value_b, const buffer_t<T> delta_b, const float lrate){
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if(k >= value_b.elem()) return;
  value_b[k] -= lrate * delta_b[k];
}

}

//! Radial Basis Function Fitting Procedure
//! 
//! We compute the current error between the data points and the radial basis support.
//! This error then gets propagated to the weights through the gradient of the RBF.
//! This is repeated for a number of steps until the error vanishes.
//!
void rbf::fit(const kdtree& kdtree, const buffer_t<vec3>& data, const size_t steps) {

  std::cout<<"Fitting RBF..."<<std::endl;

  const size_t K = this->elem;  //!< Number of RBF Centroids
  const size_t N = data.elem(); //!< Number of Fitting Data Points

  buffer_t<float> delta_w = buffer_t<float>{K, soil::host_t::GPU};

  for(int step = 0; step < steps; ++step){
    
    std::cout<<"Executing Fitting Step ("<<step<<")"<<std::endl;

    zero_delta<<<block(K, 1024), 1024>>>(delta_w);
    rbf_delta_w<<<block(N, 1024), 1024>>>(kdtree, *this, data, delta_w);
    rbf_descend<<<block(K, 1024), 1024>>>(this->weights, delta_w, this->lrate_w);

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

  // Sample Closest Points

  const size_t B = 16;
  cukd::HeapCandidateList<B> list(100.0);
  knn<B>(kdtree, pos, list);
  float val = 0.0f;

  // Closest Point Accumulation:
  for(int b = 0; b < B; ++b) {
    
    int k = list.get_pointID(b);
    if(k >= 0){
      k = kdtree.data[k].i;

      // accumulate into val!
      const vec2 c = rbf.centers[k];
      const float w = rbf.weights[k];
      const float s = rbf.shape;
      const float r = glm::length(c - pos);
      val += rbf::func(w, r, s);

    }

  }

  val_b[n] = val;

}

}
  
buffer_t<float> soil::rbf::sample(const soil::kdtree& kdtree, const buffer_t<vec2>& pos) const {

  const size_t N = pos.elem();
  auto values = soil::buffer_t<float>(N, soil::host_t::GPU);
  rbf_sample<<<block(N, 1024), 1024>>>(kdtree, *this, pos, values);
  return values;

}

}

#endif