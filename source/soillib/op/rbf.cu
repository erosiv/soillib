#ifndef SOILLIB_OP_RBF_CU
#define SOILLIB_OP_RBF_CU
#define HAS_CUDA

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

  _rbf_sample<<<block(elem, 1024), 1024>>>(this->weights, this->pos, output, pos, this->shape);
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

  _rbf_sample<<<block(elem, 1024), 1024>>>(this->weights, this->pos, output, index_t, this->shape);
  return output;

}

//
// Radial Basis Function Fitting
//

__global__ void _rbf_fit_init(buffer_t<float> weight_b, buffer_t<vec2> centroid_b, const soil::buffer_t<vec3> data_b){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= data_b.elem()) return;

  const vec3 data = data_b[n];
  weight_b[n] = 0.0f;
  centroid_b[n] = vec2(data.x, data.y);
  
}

__global__ void _rbf_fit_update(const buffer_t<float> value_b, const soil::buffer_t<vec3> data_b, buffer_t<float> weight_b, const float lrate){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= data_b.elem()) return;

  const vec3 data = data_b[n];
  const float w = weight_b[n];
  const float est = value_b[n];
  const float act = data_b[n].z;
  const float err = (est - act);//*(est - act); 
  weight_b[n] -= lrate * err;
  
}

void soil::rbf::fit(const buffer_t<vec3>& data, const size_t steps){

  const size_t elem = data.elem();

  this->weights = soil::buffer_t<float>(elem, soil::host_t::GPU);
  this->pos = soil::buffer_t<vec2>(elem, soil::host_t::GPU);
  _rbf_fit_init<<<block(elem, 1024), 1024>>>(this->weights, this->pos, data);

  for(size_t i = 0; i < steps; i++){
    const auto values = this->sample(this->pos); // evaluate at positions
    _rbf_fit_update<<<block(elem, 1024), 1024>>>(values, data, this->weights, this->lrate);
  }

}

}

#endif