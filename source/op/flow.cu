#ifndef SOILLIB_OP_FLOW
#define SOILLIB_OP_FLOW
#define HAS_CUDA

#include <soillib/op/common.hpp>
#include <soillib/op/flow.hpp>

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

//! \todo make this robust for non-regular tile shapes (test)
__device__ soil::ivec2 tile_unflatten(const unsigned int ind, const int w, const int h){

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

//
// Flow Kernel Implementation
//

template<typename T>
__global__ void _flow(soil::buffer_t<T> in, soil::buffer_t<int> out, soil::flat_t<2> index){

  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= in.elem()) return;

  const glm::ivec2 pos = index.unflatten(i);
  size_t ind = i;
//  soil::ivec2 pos = tile_unflatten(i, index[1]);
//  size_t ind = index.flatten(pos);

  T diffmax = 0.0f;
  T hvalue = in[ind];
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
    out[ind] = dirmap[value];
  else out[ind] = value;

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
// Utility Kernels
//

template<typename T>
__global__ void _fill(soil::buffer_t<T> buf, const T val){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= buf.elem()) return;
  buf[index] = val;
}

__global__ void seed(soil::buffer_t<curandState> buffer, const size_t seed, const size_t offset) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buffer.elem()) return;
  curand_init(seed, n, offset, &buffer[n]);
}

__global__ void _graph(const soil::buffer_t<glm::ivec2> in, soil::buffer_t<int> graph, soil::flat_t<2> index){

  const int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= graph.elem()) return;

  const soil::ivec2 pos = index.unflatten(ind);
  const soil::ivec2 dir = in[ind];
  
  if(index.oob(pos + dir)){
    graph[ind] = ind;
  } else {
    graph[ind] = index.flatten(pos + dir);
  }

}

__global__ void _shift(const soil::buffer_t<int> graph_a, soil::buffer_t<int> graph_b, const size_t target){

  const int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= graph_a.elem()) return;

  // we don't shift the target index,
  // because we want it 

  // each element in the graph should point to the next guy...


  // if the next cell, i.e. the one we are pointing to,
  // is the target cell, we just leave the cell as our
  // next cell, effectively keeping it static.
  const int next = graph_a[ind];
  if(next == target){
    graph_b[ind] = next;
  } else {
    graph_b[ind] = graph_a[next];
  }
}

//
// Sample Generation Procedures
//

struct sample_t {
  int index;
  float w;
};

//! Spatially Uniform Sampling:
//!   Generates a floating point value in the uniform
//!   interval [0, 1] and computes the flat buffer index. 
__device__ sample_t sample_uniform(const size_t ind, soil::flat_t<2>& index, soil::buffer_t<curandState>& randStates){

  curandState* state = &randStates[ind];
  return {
    curand_uniform(state)*index.elem(),
    float(index.elem())
  };

}

//! Streaming Resampled Importance Sampling with Reservoir Sampling:
//!
//!   Resampled reimportance sampling is implemented in a streaming manner
//!   using reservoir sampling. The suboptimal approximate distribution is
//!   the uniform distribution, which we use to generate M samples. We
//!   simultaneously utilize reservoir sampling to choose a sample with a
//!   probability that is proportional to the target distribution.
//!
//!   Note that normalization of the target distribution is not necessary,
//!   as a scaling factor can be factored out of the ration between the
//!   selected sample's weight and the sum of all weights.
//!
__device__ sample_t sample_reservoir(const size_t ind, soil::flat_t<2>& index, soil::buffer_t<curandState>& randStates, const soil::buffer_t<float>& weights){

  curandState* state = &randStates[ind];

  int sample = 0;
  float p_sample = 1.0f;
  float w_sum = 0.0f;
  const size_t M = 24;

  // Iterate over RIS Sample Count
  for(int m = 0; m < M; ++m){
   
    // Sampling Distribution: Uniform
    auto [next, w_next] = sample_uniform(ind, index, randStates);  

    // Target Distributin (Unnormalized)
    float p_target = weights[next]; // Target Distribution

    // Weighting Factor
    float w = w_next * p_target;  // Sample Weight
    w_sum += w;                   // Total Weight

    if(curand_uniform(state) <  w / w_sum){
      sample = next;
      p_sample = p_target;
    }
  
  }

  return {sample, w_sum / float(M) / p_sample};

}

//
// Accumulation Kernel Implementation
//

//! Accumulation Kernel w. Uniform Weight of 1.0
__global__ void _accumulate(const soil::buffer_t<int> graph, soil::buffer_t<float> out, soil::flat_t<2> index, soil::buffer_t<curandState> randStates, const size_t K, const size_t N){

  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if(k >= K) return;

  auto [ind, w] = sample_uniform(k, index, randStates);
  //
  int next = graph[ind];

  while(ind != next){
    ind = next;
    atomicAdd(&(out[ind]), w/float(N));
    next = graph[ind];
  }

}

//! Accumulation Kernel w. Non-Uniform Weight Buffer
__global__ void _accumulate(const soil::buffer_t<int> graph, const soil::buffer_t<float> weights, soil::buffer_t<float> out, soil::flat_t<2> index, soil::buffer_t<curandState> randStates, const size_t K, const size_t N, const bool reservoir){

  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if(k >= K) return;

  if(reservoir){

    auto [ind, w] = sample_reservoir(k, index, randStates, weights);
    int next = graph[ind];
    const float val = weights[ind];
    atomicAdd(&(out[ind]), w*val/float(N));

    while(ind != next){
      ind = next;
      next = graph[ind];
      atomicAdd(&(out[ind]), w*val/float(N));
    }
  } else {

    auto [ind, w] = sample_uniform(k, index, randStates);
    int next = graph[ind];
    const float val = weights[ind];
    atomicAdd(&(out[ind]), w*val/float(N));

    while(ind != next){
      ind = next;
      next = graph[ind];
      atomicAdd(&(out[ind]), w*val/float(N));
    }
  }

}

soil::buffer soil::accumulation(const soil::buffer& direction, const soil::index& index, int iterations, size_t samples){

  soil::select(index.type(), [&]<std::same_as<soil::flat_t<2>> I>(){});
  soil::select(direction.type(), [&]<std::same_as<soil::ivec2> T>(){});

  using I = soil::flat_t<2>;
  using T = soil::ivec2;

  auto index_t = index.as<I>();
  const size_t elem = index.elem();

  auto buffer_t = direction.as<T>();
  buffer_t.to_gpu();

  auto graph_buf = soil::buffer_t<int>{elem, soil::GPU};
  _graph<<<block(elem, 512), 512>>>(buffer_t, graph_buf, index_t);

  auto out = soil::buffer_t<float>{elem, soil::GPU};
  _fill<<<block(elem, 256), 256>>>(out, 0.0f);

  soil::buffer_t<curandState> randStates(samples, soil::host_t::GPU);
  seed<<<block(samples, 512), 512>>>(randStates, 0, 0);

  const size_t N = iterations*samples;
  
  for(int n = 0; n < iterations; ++n){
    _accumulate<<<block(samples, 512), 512>>>(graph_buf, out, index_t, randStates, samples, N);
  }

  cudaDeviceSynchronize();

  return std::move(soil::buffer(std::move(out)));

}

soil::buffer soil::accumulation(const soil::buffer& direction, const soil::buffer& weights, const soil::index& index, int iterations, size_t samples, bool reservoir){

  // Note: These will throw if not matched
  soil::select(index.type(), [&]<std::same_as<soil::flat_t<2>> I>(){});
  soil::select(direction.type(), [&]<std::same_as<soil::ivec2> T>(){});
  soil::select(weights.type(), [&]<std::same_as<float> W>(){});

  using I = soil::flat_t<2>;
  using T = soil::ivec2;
  using W = float;

  // Strict-Type Casting

  auto index_t = index.as<I>();
  const size_t elem = index.elem();

  auto buffer_t = direction.as<T>();
  buffer_t.to_gpu();

  auto weight_t = weights.as<W>();
  weight_t.to_gpu();

  // 

  auto graph_buf = soil::buffer_t<int>{elem, soil::GPU};
  _graph<<<block(elem, 512), 512>>>(buffer_t, graph_buf, index_t);

  auto out = soil::buffer_t<float>{elem, soil::GPU};
  _fill<<<block(elem, 256), 256>>>(out, 0.0f);

  soil::buffer_t<curandState> randStates(samples, soil::host_t::GPU);
  seed<<<block(samples, 512), 512>>>(randStates, 0, 0);

  const size_t N = iterations*samples;
  
  for(int n = 0; n < iterations; ++n){
    _accumulate<<<block(samples, 512), 512>>>(graph_buf, weight_t, out, index_t, randStates, samples, N, reservoir);
  }

  cudaDeviceSynchronize();

  return std::move(soil::buffer(std::move(out)));

}

//
// Exhaustive Accumulation Kernels
//

__global__ void _accumulate_exhaustive(const soil::buffer_t<int> graph, soil::buffer_t<float> out, soil::flat_t<2> index) {

  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if(k >= index.elem()) return;

  int ind = k;
  int next = graph[ind];
  atomicAdd(&(out[ind]), 1.0f);

  while(ind != next){
    ind = next;
    next = graph[ind];
    atomicAdd(&(out[ind]), 1.0f);
  }

}

__global__ void _accumulate_exhaustive(const soil::buffer_t<int> graph, const soil::buffer_t<float> weight, soil::buffer_t<float> out, soil::flat_t<2> index) {

  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if(k >= index.elem()) return;

  const float w = weight[k];
  int ind = k;
  int next = graph[ind];
  atomicAdd(&(out[ind]), w);

  while(ind != next){
    ind = next;
    next = graph[ind];
    atomicAdd(&(out[ind]), w);
  }

}

soil::buffer soil::accumulation_exhaustive(const soil::buffer& direction, const soil::index& index){

  soil::select(index.type(), [&]<std::same_as<soil::flat_t<2>> I>(){});
  soil::select(direction.type(), [&]<std::same_as<soil::ivec2> T>(){});

  using I = soil::flat_t<2>;
  using T = soil::ivec2;

  auto index_t = index.as<I>();
  const size_t elem = index.elem();

  auto buffer_t = direction.as<T>();
  buffer_t.to_gpu();

  auto graph_buf = soil::buffer_t<int>{elem, soil::GPU};
  _graph<<<block(elem, 512), 512>>>(buffer_t, graph_buf, index_t);

  auto out = soil::buffer_t<float>{elem, soil::GPU};
  _fill<<<block(elem, 256), 256>>>(out, 0.0f);

  _accumulate_exhaustive<<<block(index.elem(), 512), 512>>>(graph_buf, out, index_t);

  cudaDeviceSynchronize();

  return std::move(soil::buffer(std::move(out)));

}

soil::buffer soil::accumulation_exhaustive(const soil::buffer& direction, const soil::index& index, const soil::buffer& weights){

  // Note: These will throw if not matched
  soil::select(index.type(), [&]<std::same_as<soil::flat_t<2>> I>(){});
  soil::select(direction.type(), [&]<std::same_as<soil::ivec2> T>(){});
  soil::select(weights.type(), [&]<std::same_as<float> W>(){});

  using I = soil::flat_t<2>;
  using T = soil::ivec2;
  using W = float;

  // Strict-Type Casting

  auto index_t = index.as<I>();
  const size_t elem = index.elem();

  auto buffer_t = direction.as<T>();
  buffer_t.to_gpu();

  auto weight_t = weights.as<W>();
  weight_t.to_gpu();

  // 

  auto graph_buf = soil::buffer_t<int>{elem, soil::GPU};
  _graph<<<block(elem, 512), 512>>>(buffer_t, graph_buf, index_t);

  auto out = soil::buffer_t<float>{elem, soil::GPU};
  _fill<<<block(elem, 256), 256>>>(out, 0.0f);

  _accumulate_exhaustive<<<block(index.elem(), 512), 512>>>(graph_buf, weight_t, out, index_t);
  
  cudaDeviceSynchronize();

  return std::move(soil::buffer(std::move(out)));

}

//
// Upstream Mask Kernel Implementation
//

__global__ void _upstream(const soil::buffer_t<int> _next, soil::buffer_t<int> out, const size_t target, soil::flat_t<2> index, const size_t N){

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  soil::ivec2 pos = tile_unflatten(n, index[0], index[1]);
  size_t ind = index.flatten(pos);
  const size_t ind0 = ind;

  int found = 0;

  while(ind != target){

    int next = _next[ind];
    if(next == ind)
      break;

    ind = next;
    if(ind == target){
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
      const size_t target_index = index_t.flatten(target);

      auto graph_buf_a = soil::buffer_t<int>{elem, soil::GPU};
      auto graph_buf_b = soil::buffer_t<int>{elem, soil::GPU};
      _graph<<<block(elem, 512), 512>>>(buffer_t, graph_buf_a, index_t);
      _shift<<<block(elem, 512), 512>>>(graph_buf_a, graph_buf_b, target_index);
      _shift<<<block(elem, 512), 512>>>(graph_buf_b, graph_buf_a, target_index);
      _shift<<<block(elem, 512), 512>>>(graph_buf_a, graph_buf_b, target_index);
      _shift<<<block(elem, 512), 512>>>(graph_buf_b, graph_buf_a, target_index);
      _shift<<<block(elem, 512), 512>>>(graph_buf_a, graph_buf_b, target_index);
      _shift<<<block(elem, 512), 512>>>(graph_buf_b, graph_buf_a, target_index);

      auto out = soil::buffer_t<int>{elem, soil::GPU};
      _fill<<<block(elem, 256), 256>>>(out, 0);

      if(!index_t.oob(target)){
        _upstream<<<block(elem, 512), 512>>>(graph_buf_a, out, target_index, index_t, elem);
      }
      cudaDeviceSynchronize();

      return std::move(soil::buffer(std::move(out)));

    });

  });

}

//
// Upstream Distance Kernel Implementation
//

__global__ void _distance(soil::buffer_t<int> _next, soil::buffer_t<int> out, const size_t target, soil::flat_t<2> index, const size_t N){

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  soil::ivec2 pos = tile_unflatten(n, index[0], index[1]);
  size_t ind = index.flatten(pos);
  const size_t ind0 = ind;

  // note: upper bound is absolute worst-case scenario
  for(int step = 0; step < N; ++step){

    if(ind == target){
      out[ind0] = step;
      break;
    }

    int next = _next[ind];
    if(next == ind)
      break;
 
    ind = next;

  }

}

soil::buffer soil::distance(const soil::buffer& buffer, const soil::index& index, const glm::ivec2 target){

  return soil::select(index.type(), [&]<std::same_as<soil::flat_t<2>> I>() {
    return soil::select(buffer.type(), [&]<std::same_as<soil::ivec2> T>(){

      auto index_t = index.as<I>();
      auto buffer_t = buffer.as<T>();
      buffer_t.to_gpu();

      const size_t elem = index.elem();
      const size_t target_index = index_t.flatten(target);

      auto out = soil::buffer_t<int>{elem, soil::GPU};

      auto graph_buf_a = soil::buffer_t<int>{elem, soil::GPU};
      auto graph_buf_b = soil::buffer_t<int>{elem, soil::GPU};
      _graph<<<block(elem, 512), 512>>>(buffer_t, graph_buf_a, index_t);

      //!\todo figure out how many iterations of this are necessary?
      //!\todo fix the scale of the steps output, which is currently incorrect
      //! because multiple steps have been merged together
      _shift<<<block(elem, 512), 512>>>(graph_buf_a, graph_buf_b, target_index);
      _shift<<<block(elem, 512), 512>>>(graph_buf_b, graph_buf_a, target_index);
      _shift<<<block(elem, 512), 512>>>(graph_buf_a, graph_buf_b, target_index);
      _shift<<<block(elem, 512), 512>>>(graph_buf_b, graph_buf_a, target_index);
      _shift<<<block(elem, 512), 512>>>(graph_buf_a, graph_buf_b, target_index);
      _shift<<<block(elem, 512), 512>>>(graph_buf_b, graph_buf_a, target_index);

      _fill<<<block(elem, 256), 256>>>(out, -1); // unknown state...
      if(!index_t.oob(target)){
        _distance<<<block(elem, 512), 512>>>(graph_buf_a, out, target_index, index_t, elem);
      }
      cudaDeviceSynchronize();

      return std::move(soil::buffer(std::move(out)));

    });
  });

}

// note: move this to a different file
#endif