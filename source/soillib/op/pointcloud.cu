#ifndef SOILLIB_OP_POINTCLOUD
#define SOILLIB_OP_POINTCLOUD
#define HAS_CUDA

#include <soillib/soillib.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <curand_kernel.h>
#include <soillib/op/common.hpp>
#include <soillib/op/gather.hpp>

namespace soil {

namespace {
  __global__ void seed(buffer_t<curandState> buffer, const size_t seed, const size_t offset) {
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= buffer.elem()) return;
    curand_init(seed, n, offset, &buffer[n]);
  }
}

//template<typename T, typename Index, typename Flat>
__global__ void _sample_pointcloud(soil::buffer_t<float> input, soil::buffer_t<vec3> output, soil::buffer_t<curandState> rand, const soil::flat_t<2> index, const size_t N){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  curandState* randState = &rand[n];
  vec2 pos = vec2{
    curand_uniform(randState)*float(index[0]),
    curand_uniform(randState)*float(index[1])
  };
  
  lerp_t<float> lerp = gather(input, index, pos);
  while(isnan(lerp.val())){
    pos = vec2{
      curand_uniform(randState)*float(index[0]),
      curand_uniform(randState)*float(index[1])
    };
    lerp = gather(input, index, pos);
  }
  output[n] = vec3(pos.x, pos.y, lerp.val());

}

soil::buffer_t<vec3> sample_pointcloud_impl(const soil::buffer_t<float> &buffer, const soil::index &index, const size_t N){

  soil::buffer_t<vec3> output(N, soil::GPU);
  soil::buffer_t<curandState> rand(N, soil::host_t::GPU);

  const auto index_t = index.as<soil::flat_t<2>>();
  seed<<<block(N, 1024), 1024>>>(rand, 0, 0);
  cudaDeviceSynchronize();
  _sample_pointcloud<<<block(N, 1024), 1024>>>(buffer, output, rand, index_t, N);

  return output;

}

}

#endif