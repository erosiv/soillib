#define HAS_CUDA

#include <soillib/node/flow.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>
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

__global__ void _direction(soil::buffer_t<int> in, soil::buffer_t<glm::ivec2> out){

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

template<typename T>
__global__ void _fill(soil::buffer_t<T> buf, const T val){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < buf.size())
    buf[index] = val;
}

__global__ void init_randstate(curandState* states, const size_t N, const size_t seed) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= N) return;
  curand_init(seed, index, 0, &states[index]);
}

__global__ void _accumulate(soil::buffer_t<glm::ivec2> in, soil::buffer_t<int> out, soil::flat_t<2> index, curandState* randStates, const int steps, const int N){

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  curandState* state = &randStates[n];
  glm::ivec2 pos {
    curand_uniform(state)*index[0],
    curand_uniform(state)*index[1]
  };
  size_t ind = index.flatten(pos);

  for(size_t s = 0; s < steps; ++s){

    const glm::ivec2 dir = in[ind];
    pos += dir;
    if(dir[0] == 0 && dir[1] == 0)
      break;

    if(index.oob(pos))
      break;

    ind = index.flatten(pos);
    atomicAdd(&(out[ind]), 1);
  }

}

__global__ void _normalize(soil::buffer_t<int> in, soil::buffer_t<double> out, double P){
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= in.elem()) return;
  out[n] = 1.0 + P * (double)in[n];
}

soil::buffer soil::flow::full() const {

  const int elem = index.elem();
  auto in = this->buffer.as<double>();
  in.to_gpu();

  const int thread = 1024;
  const int block = (elem + thread - 1)/thread;
  
  auto out = buffer_t<int>{index.elem(), GPU};
  _flow<<<block, thread>>>(in, out, index);

  return std::move(soil::buffer(std::move(out)));

}

soil::buffer soil::direction::full() const {

  const int elem = index.elem();
  auto in = this->buffer.as<int>();
  in.to_gpu();

  const int thread = 1024;
  const int block = (elem + thread - 1)/thread;

  auto out = buffer_t<ivec2>{index.elem(), GPU};
  _direction<<<block, thread>>>(in, out);

  return std::move(soil::buffer(std::move(out)));

}

__global__ void _upstream(soil::buffer_t<glm::ivec2> in, soil::buffer_t<int> out, glm::ivec2 target, soil::flat_t<2> index, const size_t N){

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  bool found = false;
  size_t ind = n;
  glm::ivec2 pos =  index.unflatten(n);
  size_t target_ind = index.flatten(target);

  // note: upper bound is absolute worst-case scenario
  for(int step = 0; step < N; ++step){

    if(ind == target_ind){
      found = true;
      break;
    }

    const glm::ivec2 dir = in[ind];
    pos += dir;
    if(dir[0] == 0 && dir[1] == 0)
      break;

    if(index.oob(pos))
      break;

    ind = index.flatten(pos);

  }

  if(found)
    out[n] = 1;
  else out[n] = 0;

}

soil::buffer soil::upstream::full() const {

  // I suppose the ideal solution is to do a random order...
  // how do we generate a list of random order?
  // we could do a perfect hash instead...
  // or we could really just use a random number generator
  // and speculate that we get sufficient hits.
  // this determinism is not necessarily suited to GPU computation...
  // unless I can perform some kind of radix sort on the elements
  // but that would require some kind of tree index which I might not have.
  // the sort would basically be: am I above or below a value...
  // but we would sort over the set of indices...

  /*
  so if we did a sort that would effectively yield the solution...
  for now, we will just do it in order and see if shuffling improves performance at all later...
  */

  // Input Direction Buffer!
  const size_t elem = index.elem();
  auto in = this->buffer.as<ivec2>();
  in.to_gpu();

  auto out = buffer_t<int>{elem, GPU};
  int thread = 1024;
  int block = (elem + thread - 1)/thread;
  _fill<<<block, thread>>>(out, 2); // unknown state...
  
  thread = 1024;
  block = (elem + thread - 1)/thread;
  _upstream<<<block, thread>>>(in, out, target, index, elem);

  return std::move(soil::buffer(std::move(out)));

}

soil::buffer soil::accumulation::full() const {

  const size_t elem = index.elem();
  auto in = this->buffer.as<ivec2>();
  in.to_gpu();

  auto out = buffer_t<int>{elem, GPU};
  int thread = 1024;
  int block = (elem + thread - 1)/thread;
  _fill<<<block, thread>>>(out, 0);

  auto out2 = buffer_t<double>{elem, GPU};
  
  curandState* randStates;
  cudaMalloc((void**)&randStates, this->samples * sizeof(curandState));

  thread = 1024;
  block = (this->samples + thread - 1)/thread;

  init_randstate<<<block, thread>>>(randStates, this->samples, 0);

  for(int n = 0; n < this->iterations; ++n)
    _accumulate<<<block, thread>>>(in, out, index, randStates, this->steps, this->samples);
  cudaFree(randStates);

  thread = 1024;
  block = (elem + thread - 1)/thread;
  const double P = double(elem)/double(iterations*samples);
  _normalize<<<block, thread>>>(out, out2, P);

  return std::move(soil::buffer(std::move(out2)));

}

// note: move this to a different file
