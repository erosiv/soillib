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

//
// Uniform Pointcloud Sampling
//

__global__ void _pointcloud_sample(soil::buffer_t<float> input, soil::buffer_t<vec3> output, soil::buffer_t<curandState> rand, const soil::flat_t<2> index, const size_t N){

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

soil::buffer_t<vec3> pointcloud_sample_impl(const soil::buffer_t<float> &buffer, const soil::index &index, const size_t N){

  soil::buffer_t<vec3> output(N, soil::GPU);
  soil::buffer_t<curandState> rand(N, soil::host_t::GPU);

  const auto index_t = index.as<soil::flat_t<2>>();
  seed<<<block(N, 1024), 1024>>>(rand, 0, 0);
  cudaDeviceSynchronize();
  _pointcloud_sample<<<block(N, 1024), 1024>>>(buffer, output, rand, index_t, N);

  return output;

}

//
// Pointcloud Scaling
//

__global__ void _pointcloud_scale(soil::buffer_t<vec3> input, const soil::flat_t<2> index, const soil::vec3 scale){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= input.elem()) return;

  const vec3 wmin = vec3(0.0f, 0.0f, -1.0f) * scale;
  const vec3 wmax = vec3(index[0], index[1], 1.0f) * scale;
  const vec3 wext = (wmax - wmin);      // Extent of Buffer in World-Space
  const vec3 wmid = (wmax + wmin)*0.5f; // Center Point in World-Space
  const float wscale = glm::max(wext.x, glm::max(wext.y, wext.z));

  vec3 pos = input[n] * scale;
  vec3 cpos = 2.0f*(pos - wmid)/wscale;
  input[n] = cpos;

}

void pointcloud_scale_impl(const soil::buffer_t<vec3> &buffer, const soil::index &index, const soil::vec3 scale){

  const auto index_t = index.as<soil::flat_t<2>>();
  _pointcloud_scale<<<block(buffer.elem(), 1024), 1024>>>(buffer, index_t, scale);

}

}

#endif