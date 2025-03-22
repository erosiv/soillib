#ifndef SOILLIB_OP_RBF_CU
#define SOILLIB_OP_RBF_CU
#define HAS_CUDA

#include <curand_kernel.h>
#include <soillib/op/rbf.hpp>
#include <soillib/op/common.hpp>

// Radial Basis Function Interpolator
// 1. Fit the Weight-Set from a set of Values, Positions
//  1.1 For Different Interpolator Types
//  1.2 Allow for Shape-Parameter Optimization as well?
// 2. Compute the Values from a Weight-Set, Positions
//  Kernelize this for Performance
//  Obviously, implement the gradient of this as well.

namespace soil {

//
// Initialize RBF Centroids
//

namespace {

__global__ void seed(buffer_t<curandState> buffer, const size_t seed, const size_t offset) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buffer.elem()) return;
  curand_init(seed, n, offset, &buffer[n]);
}

__global__ void rbf_init(buffer_t<float> weight_b, buffer_t<vec2> center_b, const soil::buffer_t<vec3> data_b){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= data_b.elem()) return;
  const vec3 data = data_b[n];
  weight_b[n] = 0.0f;
  center_b[n] = vec2(data.x, data.y);
}

__global__ void rbf_init(buffer_t<float> weight_b, buffer_t<vec2> center_b, buffer_t<curandState> rand, const flat_t<2> index){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= rand.elem()) return;
  curandState* randState = &rand[n];
  weight_b[n] = 0.0f;
  center_b[n] = vec2 {
    curand_uniform(randState)*float(index[0]-1),
    curand_uniform(randState)*float(index[1]-1)
  };
}

}

void rbf::init(const buffer_t<vec3>& data){

  const size_t elem = data.elem();
  this->elem = elem;

  this->weights = soil::buffer_t<float>(elem, soil::host_t::GPU);
  this->centers = soil::buffer_t<vec2>(elem, soil::host_t::GPU);

  rbf_init<<<block(elem, 1024), 1024>>>(this->weights, this->centers, data);

}

void rbf::init(const index& index, const size_t N){

  const size_t elem = N;
  this->elem = elem;

  this->weights = soil::buffer_t<float>(elem, soil::host_t::GPU);
  this->centers = soil::buffer_t<vec2>(elem, soil::host_t::GPU);

  auto rand = soil::buffer_t<curandState>(elem, soil::host_t::GPU);
  seed<<<block(N, 1024), 1024>>>(rand, 1, 2);
  cudaDeviceSynchronize();

  auto index_t = index.as<flat_t<2>>();
  rbf_init<<<block(elem, 1024), 1024>>>(this->weights, this->centers, rand, index_t);

}

//! Sample Function Implementation
//!  Note: This utilizes all points, when in reality
//!  only points within a certain radius have to be
//!  considered for a certain amount of accuracy.
//!  This requires implementation of an acceleration structure.
//
__device__ float _rbf_sample(const buffer_t<float>& w, const buffer_t<vec2>& c, const vec2 p, const float shape){
  float val = 0.0f;
  for(int k = 0; k < w.elem(); ++k){
    val += w[k] * rbf::func(glm::length(c[k] - p), shape);
  }
  return val;
}

//
// Simple Radial Basis Function Sampling
//

// Position-Buffer based Sampling

__global__ void _rbf_sample(const buffer_t<float> weight_b, const buffer_t<vec2> centroid_b, soil::buffer_t<float> val_b, const soil::buffer_t<vec2> pos, const float shape){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= pos.elem()) return;
  val_b[n] = _rbf_sample(weight_b, centroid_b, pos[n], shape);
}

buffer_t<float> soil::rbf::sample(const buffer_t<vec2>& pos) const {

  auto output = soil::buffer_t<float>(pos.elem(), soil::host_t::GPU);
  const size_t elem = pos.elem();

  _rbf_sample<<<block(elem, 1024), 1024>>>(this->weights, this->centers, output, pos, this->shape);
  return output;

}

// Index Based Sampling (Full Shape)

__global__ void _rbf_sample(const buffer_t<float> weight_b, const buffer_t<vec2> centroid_b, soil::buffer_t<float> val_b, const soil::flat_t<2> index, const float shape){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= index.elem()) return;
  val_b[n] = _rbf_sample(weight_b, centroid_b, index.unflatten(n), shape);
}

buffer_t<float> soil::rbf::sample(const index& index) const {

  const auto index_t = index.as<soil::flat_t<2>>();
  auto output = soil::buffer_t<float>(index.elem(), soil::host_t::GPU);
  const size_t elem = index_t.elem();

  _rbf_sample<<<block(elem, 1024), 1024>>>(this->weights, this->centers, output, index_t, this->shape);
  return output;

}

//
// Radial Basis Function Fitting
//  Note: Gradient Descent on L1 Metric
//

namespace {

__global__ void rbf_matrix(const flat_t<2> shape, const buffer_t<vec2> center_b, const soil::buffer_t<vec3> data_b, buffer_t<float> matrix_b, const float shapef){

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= matrix_b.elem()) return;

  const ivec2 ind = shape.unflatten(i);
  const size_t n = ind[0];
  const size_t k = ind[1];
  
  const vec3 data = data_b[n];
  const vec2 ctr = center_b[k];
  const vec2 pos = vec2(data.x, data.y);

  matrix_b[i] = rbf::func(glm::length(ctr - pos), shapef);

}

//! Launch with the Number of Components in out_b
__global__ void rbf_matvec(const flat_t<2> shape, const buffer_t<float> matrix_b, const buffer_t<float> vector_b, buffer_t<float> out_b, const soil::buffer_t<vec3> data_b){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= out_b.elem()) return;

  const size_t N = shape[0];
  const size_t K = shape[1];

  float val = 0.0f;
  for(int k = 0; k < K; ++k){
    const int i = shape.flatten(ivec2(n, k));
    val += vector_b[k] * matrix_b[i];
  }

  out_b[n] = val - data_b[n].z;

}

__global__ void rbf_matvecT(const flat_t<2> shape, const buffer_t<float> matrix_b, const buffer_t<float> vector_b, buffer_t<float> out_b){

  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if(k >= out_b.elem()) return;

  const size_t N = shape[0];
  const size_t K = shape[1];

  float val = 0.0f;
  for(int n = 0; n < N; ++n){
    const int i = shape.flatten(ivec2(n, k));
    val += vector_b[n] * matrix_b[i];
  }
  
  out_b[k] = val;

}

__global__ void rbf_descend(buffer_t<float> value_b, const buffer_t<float> delta_b, const float lrate){
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if(k >= value_b.elem()) return;
  value_b[k] -= lrate * delta_b[k];
}

}

void soil::rbf::fit(const buffer_t<vec3>& data, const size_t steps){

  const size_t N = data.elem();
  const size_t K = this->elem;
  const flat_t<2> shape({N, K});

  auto matrix = soil::buffer_t<float>(shape.elem(), soil::host_t::GPU);
  auto adiff = soil::buffer_t<float>(N, soil::host_t::GPU); // Approximation Difference
  auto delta = soil::buffer_t<float>(K, soil::host_t::GPU); // Descent Delta
  rbf_matrix<<<block(shape.elem(), 1024), 1024>>>(shape, this->centers, data, matrix, this->shape);

  for(size_t i = 0; i < steps; i++){

    rbf_matvec<<<block(N, 1024), 1024>>>(shape, matrix, this->weights, adiff, data);
    rbf_matvecT<<<block(K, 1024), 1024>>>(shape, matrix, adiff, delta);
    rbf_descend<<<block(elem, 1024), 1024>>>(this->weights, delta, this->lrate);
  
  }

}

}

#endif