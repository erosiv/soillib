#ifndef SOILLIB_OP_POINTCLOUD
#define SOILLIB_OP_POINTCLOUD
#define HAS_CUDA

#include <soillib/soillib.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <curand_kernel.h>
#include <soillib/op/common.hpp>
#include <soillib/op/gather.hpp>

#include <soillib/op/pointcloud.hpp>

namespace soil {

namespace {

__global__ void seed(buffer_t<curandState> buffer, const size_t seed, const size_t offset) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buffer.elem()) return;

  curand_init(seed, n, offset, &buffer[n]);

}

}

//
// Random Position Sampling within Index Bound
//

__global__ void _sample_N(soil::buffer_t<vec2> output, soil::buffer_t<curandState> rand, const soil::flat_t<2> index, const size_t N){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  curandState* randState = &rand[n];
  output[n] = vec2 {
    curand_uniform(randState)*float(index[0]-1),
    curand_uniform(randState)*float(index[1]-1)
  };

}

soil::buffer_t<vec2> sample_N_impl(const soil::flat_t<2> &index, const size_t N){

  soil::buffer_t<vec2> output(N, soil::GPU);
  soil::buffer_t<curandState> rand(N, soil::host_t::GPU);

  seed<<<block(N, 1024), 1024>>>(rand, 0, 0);
  cudaDeviceSynchronize();
  _sample_N<<<block(N, 1024), 1024>>>(output, rand, index, N);

  return output;

}

//
// Sample Lerp Implementation
//

__global__ void _sample_lerp(const soil::buffer_t<float> field, soil::buffer_t<float> output, const soil::flat_t<2> index, const soil::buffer_t<vec2> pos_b){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= pos_b.elem()) return;

  vec2 pos = pos_b[n];
  
  lerp_t<float> lerp = gather(field, index, pos);
  output[n] = lerp.val();

}

soil::buffer_t<float> sample_lerp_impl(const soil::buffer_t<float> &field, const soil::flat_t<2> &index, const soil::buffer_t<vec2>& pos){

  const size_t elem = pos.elem();
  soil::buffer_t<float> output(elem, soil::GPU);
  _sample_lerp<<<block(elem, 1024), 1024>>>(field, output, index, pos);

  return output;

}

//
// Uniform Pointcloud Sampling
//

__global__ void _pointcloud_sample(soil::buffer_t<float> input, soil::buffer_t<vec3> output, soil::buffer_t<curandState> rand, const soil::flat_t<2> index, const size_t N){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  curandState* randState = &rand[n];
  vec2 pos = vec2{
    curand_uniform(randState)*float(index[0]-1),
    curand_uniform(randState)*float(index[1]-1)
  };
  
  lerp_t<float> lerp = gather(input, index, pos);
  while(isnan(lerp.val())){
    pos = vec2{
      curand_uniform(randState)*float(index[0]-1),
      curand_uniform(randState)*float(index[1]-1)
    };
    lerp = gather(input, index, pos);
  }
  output[n] = vec3(pos.x, pos.y, lerp.val());

}

soil::buffer_t<vec3> pointcloud_sample_impl(const soil::buffer_t<float> &buffer, const soil::index &index, const size_t N){

  soil::buffer_t<vec3> output(N, soil::GPU);
  soil::buffer_t<curandState> rand(N, soil::host_t::GPU);

  const auto index_t = index.as<soil::flat_t<2>>();
  seed<<<block(N, 1024), 1024>>>(rand, 0, 0);
  cudaDeviceSynchronize();
  _pointcloud_sample<<<block(N, 1024), 1024>>>(buffer, output, rand, index_t, N);

  return output;

}

} // end of namespace soil

#endif