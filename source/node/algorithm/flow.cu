#define HAS_CUDA

#include <soillib/node/algorithm/flow.hpp>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <iostream>
#include <glm/glm.hpp>

namespace {

__device__ const glm::ivec2 coords[8] = {
  glm::ivec2{-1, 0},
  glm::ivec2{-1, 1},
  glm::ivec2{ 0, 1},
  glm::ivec2{ 1, 1},
  glm::ivec2{ 1, 0},
  glm::ivec2{ 1,-1},
  glm::ivec2{ 0,-1},
  glm::ivec2{-1,-1},
};

__device__ const double dist[8] = {
  1.0,
  CUDART_SQRT_TWO,
  1.0,
  CUDART_SQRT_TWO,
  1.0,
  CUDART_SQRT_TWO,
  1.0,
  CUDART_SQRT_TWO
};

__device__ const int dirmap[8] = {
  7, 8, 1, 2, 3, 4, 5, 6,
};

}

__global__ void _flow(soil::buffer_t<double> in, soil::buffer_t<int> out, soil::flat_t<2> index){

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= in.elem()) return;

  const glm::ivec2 pos = index.unflatten(i);
    
  double diffmax = 0.0f;
  double hvalue = in[i];
  int value = -2;   // default also for nan
  bool pit = true;
  bool has_flow = false;

  for(size_t k = 0; k < 8; ++k){

    const glm::ivec2 coord = coords[k];
    const glm::ivec2 npos = pos + coord;

    if(!index.oob(npos)){
      
      const size_t n = index.flatten(npos);
      const double nvalue = in[n];
      const double ndiff = (hvalue - nvalue)/dist[k];
      
      if(ndiff > diffmax){
        value = k;
        diffmax = ndiff;
      }

      has_flow |= (ndiff > 0.0);
      pit &= (ndiff < 0.0);

      // note: equivalent
      // if(ndiff > 0.0) has_flow = true;
      // if(ndiff >= 0.0) pit = false;

    }

  }

  if(pit) value = -2;
  if(!has_flow && !pit) value = -1;

  if(value >= 0)
    out[i] = dirmap[value];
  else out[i] = value;

}

__global__ void _select(soil::buffer_t<int> in, soil::buffer_t<glm::ivec2> out){

  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= in.elem()) return;

  glm::ivec2 val(0, 0);
  for(size_t k = 0; k < 8; ++k){
    if(in[index] == dirmap[k]){
      val = coords[k];
      break;
    }
  }

  out[index] = val;

}

soil::buffer soil::flow::full() const {

  const int elem = index.elem();
  auto in = this->buffer.as<double>();
  auto out = buffer_t<int>{index.elem(), GPU};
  in.to_gpu();

  const int thread = 1024;
  const int block = (elem + thread - 1)/thread;
  _flow<<<block, thread>>>(in, out, index);

  return std::move(soil::buffer(std::move(out)));

}

soil::buffer soil::direction::full() const {

  const int elem = index.elem();
  auto in = this->buffer.as<int>();
  auto out = buffer_t<ivec2>{elem, GPU};
  in.to_gpu();

  const int thread = 1024;
  const int block = (elem + thread - 1)/thread;
  _select<<<block, thread>>>(in, out);

  out.to_cpu();

  return std::move(soil::buffer(std::move(out)));

}













soil::buffer soil::accumulation::full() const {

  const size_t elem = index.elem();
  auto in = this->buffer.as<ivec2>();
  auto out = buffer_t<double>{elem};

  const double P = double(elem)/double(iterations*samples);

  for(size_t i = 0; i < elem; ++i)
    out[i] = 0.0;

  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist_x(0, index[0]-1);
  std::uniform_int_distribution<std::mt19937::result_type> dist_y(0, index[1]-1);

  for(size_t i = 0; i < iterations; ++i){
    for(size_t n = 0; n < samples; ++n){

      ivec2 pos{dist_x(rng), dist_y(rng)};
      size_t ind = index.flatten(pos);

      for(size_t s = 0; s < steps; ++s){

        const ivec2 dir = in[ind];
        pos += dir;
        if(dir[0] == 0 && dir[1] == 0)
          break;

        if(index.oob(pos))
          break;

        ind = index.flatten(pos);
        out[ind] += 1.0;

      }
    }
  }

  // Note:
  // We could techincally also accumulate P,
  // then add 1. For some reason, slower.

  for(size_t i = 0; i < elem; i++){
    out[i] = 1.0 + P*out[i];
  }

  return std::move(soil::buffer(std::move(out)));

}