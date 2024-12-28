#define HAS_CUDA

#include <soillib/node/flow.hpp>
#include <soillib/core/texture.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include <iostream>
#include <glm/glm.hpp>

/*
Potential Concept for Improving Performance:
Instead of letting each possible path run in a batch
until completion, we should run then for a fixed number
of steps and then check which paths have not yet terminated.

Those that have not yet terminated are re-entered into a queue
and we re-execute for only those guys... That should basically
give an overall performance boost.
*/

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

int block(const int elem, const int thread){
  return (elem + thread - 1)/thread;
}

}

//
// Flow Kernel Implementation
//

template<typename T>
__global__ void _flow(soil::buffer_t<T> in, soil::buffer_t<int> out, soil::flat_t<2> index){

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= in.elem()) return;

  const glm::ivec2 pos = index.unflatten(i);
    
  T diffmax = 0.0f;
  T hvalue = in[i];
  int value = -2;   // default also for nan
  bool pit = true;
  bool has_flow = false;

  for(size_t k = 0; k < 8; ++k){

    const glm::ivec2 coord = coords[k];
    const glm::ivec2 npos = pos + coord;

    if(!index.oob(npos)){
      
      const size_t n = index.flatten(npos);
      const T nvalue = in[n];
      const T ndiff = (hvalue - nvalue)/T(dist[k]);
      
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

soil::buffer soil::flow(const soil::buffer& buffer, const soil::index& index) {

  return soil::select(index.type(), [&]<std::same_as<soil::flat_t<2>> I>() {
    return soil::select(buffer.type(), [&]<std::floating_point T>(){

      auto index_t = index.as<I>();
      auto buffer_t = buffer.as<T>();
      buffer_t.to_gpu();

      const int elem = index_t.elem();
      auto out = soil::buffer_t<int>{index_t.elem(), soil::GPU};

      _flow<<<block(elem, 256), 256>>>(buffer_t, out, index_t);
      cudaDeviceSynchronize();

      return std::move(soil::buffer(std::move(out)));

    });

  });

}

//
// Direction Kernel Implementation
//

__device__ soil::ivec2 _get_dir(const int flow){
  for(size_t k = 0; k < 8; ++k){
    if(flow == dirmap[k])
      return coords[k];
  }
  return {0, 0};
}

__global__ void _direction(soil::buffer_t<int> in, soil::buffer_t<glm::ivec2> out){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= in.elem()) return;
  out[index] = _get_dir(in[index]);
}

soil::buffer soil::direction(const soil::buffer& buffer, const soil::index& index){

  return soil::select(index.type(), [&]<std::same_as<soil::flat_t<2>> I>() {
    return soil::select(buffer.type(), [&]<std::same_as<int> T>(){

      auto index_t = index.as<I>();
      auto buffer_t = buffer.as<T>();
      buffer_t.to_gpu();

      const int elem = index_t.elem();
      auto out = soil::buffer_t<soil::ivec2>{index_t.elem(), soil::GPU};

      _direction<<<block(elem, 256), 256>>>(buffer_t, out);
      cudaDeviceSynchronize();

      return std::move(soil::buffer(std::move(out)));

    });

  });

}

//
// Accumulation Kernel Implementation
//

template<typename T>
__global__ void _fill(soil::buffer_t<T> buf, const T val){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= buf.elem()) return;
  buf[index] = val;
}

__global__ void init_randstate(curandState* states, const size_t N, const size_t seed) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= N) return;
  curand_init(seed, index, 0, &states[index]);
}

__global__ void _accumulate(soil::texture<int> texture, soil::buffer_t<int> out, soil::flat_t<2> index, curandState* randStates, const int steps, const int N){

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  curandState* state = &randStates[n];
  size_t ind = curand_uniform(state)*index.elem();
  glm::ivec2 pos = index.unflatten(ind);

  for(int s = 0; s < steps; ++s){

    const int flow = texture[soil::vec2(pos)];
    const glm::ivec2 dir = _get_dir(flow);
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

soil::buffer soil::accumulation(const soil::buffer& buffer, const soil::index& index, int iterations, int samples, int steps){

  return soil::select(index.type(), [&]<std::same_as<soil::flat_t<2>> I>() {
    return soil::select(buffer.type(), [&]<std::same_as<int> T>(){

      auto index_t = index.as<I>();
      auto buffer_t = buffer.as<T>();
      buffer_t.to_gpu();

      texture<int> texture(buffer_t, index_t);

      const size_t elem = index.elem();
      auto out = soil::buffer_t<int>{elem, soil::GPU};
      auto out2 = soil::buffer_t<double>{elem, soil::GPU};

      _fill<<<block(elem, 256), 256>>>(out, 0);

      curandState* randStates;
      cudaMalloc((void**)&randStates, samples * sizeof(curandState));
      init_randstate<<<block(samples, 256), 256>>>(randStates, samples, 0);

      for(int n = 0; n < iterations; ++n)
        _accumulate<<<block(samples, 512), 512>>>(texture, out, index_t, randStates, steps, samples);

      const double P = double(elem)/double(iterations*samples);
      _normalize<<<block(elem, 256), 256>>>(out, out2, P);

      cudaFree(randStates);
      cudaDeviceSynchronize();

      return std::move(soil::buffer(std::move(out2)));

    });

  });

}

//
// Upstream Mask Kernel Implementation
//



namespace {

//! \todo make this robust for non-regular tile shapes (test)
__device__ soil::ivec2 tile_unflatten(const unsigned int ind, const int h){

  constexpr int tile_w = 8;
  constexpr int tile_h = 8;
  constexpr int tile_s = tile_w * tile_h;

  // Binned Tile Index, Tile Position

  unsigned int tile_ind = ind / tile_s;
  unsigned int tile_x = tile_w * (tile_ind / (h / tile_h));
  unsigned int tile_y = tile_h * (tile_ind % (h / tile_h));

  unsigned int tile_pos = ind % tile_s;
  unsigned int x = tile_x + tile_pos / tile_h;
  unsigned int y = tile_y + tile_pos % tile_h;

  return soil::ivec2(x, y);

}

}

__global__ void _upstream(const soil::buffer_t<glm::ivec2> in, soil::buffer_t<int> out, glm::ivec2 target, soil::flat_t<2> index, const size_t N){

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  soil::ivec2 pos = tile_unflatten(n, index[0]);
  size_t ind = index.flatten(pos);
  const size_t ind0 = ind;
  soil::ivec2 dir;

  int found = 0;

  size_t target_ind = index.flatten(target);

  // note: upper bound is absolute worst-case scenario
  while(ind != target_ind && !index.oob(pos)){

    dir = in[ind];
    if(dir[0] == 0 && dir[1] == 0)
      break;

    pos += dir;
    ind = index.flatten(pos);
    if(ind == target_ind){
      found = 1;
    }

  }

  out[ind0] |= found;
}

// Note: This can potentially be made faster, by batching the upstream kernel execution
// and testing against positions tested in the previous batch (using the output buffer)
// This could be done using a shuffled index buffer (e.g. perfect hash), or using some
// other regular permuation to improve performance. This is not guaranteed to be better.
soil::buffer soil::upstream(const soil::buffer& buffer, const soil::index& index, const glm::ivec2 target){

  return soil::select(index.type(), [&]<std::same_as<soil::flat_t<2>> I>() {
    return soil::select(buffer.type(), [&]<std::same_as<soil::ivec2> T>(){

      auto index_t = index.as<I>();
      auto buffer_t = buffer.as<soil::ivec2>();
      buffer_t.to_gpu();

      const size_t elem = index_t.elem();
      auto out = soil::buffer_t<int>{elem, soil::GPU};

      _fill<<<block(elem, 256), 256>>>(out, 0);
      if(!index_t.oob(target)){
        _upstream<<<block(elem, 512), 512>>>(buffer_t, out, target, index_t, elem);
      }
      cudaDeviceSynchronize();

      return std::move(soil::buffer(std::move(out)));

    });

  });

}

//
// Upstream Distance Kernel Implementation
//

__global__ void _distance(soil::buffer_t<glm::ivec2> in, soil::buffer_t<int> out, glm::ivec2 target, soil::flat_t<2> index, const size_t N){

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  soil::ivec2 pos = tile_unflatten(n, index[0]);
  size_t ind = index.flatten(pos);
  const size_t ind0 = ind;
  soil::ivec2 dir;

  size_t target_ind = index.flatten(target);

  // note: upper bound is absolute worst-case scenario
  for(int step = 0; step < N; ++step){

    if(ind == target_ind){
      out[ind0] = step;
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

}

soil::buffer soil::distance(const soil::buffer& buffer, const soil::index& index, const glm::ivec2 target){

  return soil::select(index.type(), [&]<std::same_as<soil::flat_t<2>> I>() {
    return soil::select(buffer.type(), [&]<std::same_as<soil::ivec2> T>(){

      auto index_t = index.as<I>();
      auto buffer_t = buffer.as<T>();
      buffer_t.to_gpu();

      const size_t elem = index.elem();
      auto out = soil::buffer_t<int>{elem, soil::GPU};

      _fill<<<block(elem, 256), 256>>>(out, -1); // unknown state...
      if(!index_t.oob(target)){
        _distance<<<block(elem, 512), 512>>>(buffer_t, out, target, index_t, elem);
      }
      cudaDeviceSynchronize();

      return std::move(soil::buffer(std::move(out)));

    });
  });

}

// note: move this to a different file