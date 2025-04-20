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
// Buffer Concatenation
//

__global__ void _concat(soil::buffer_t<vec3> output, const soil::buffer_t<vec2> a_t, const soil::buffer_t<float> b_t){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= output.elem()) return;

  vec2 a = a_t[n];
  float b = b_t[n];
  output[n] = vec3(a.x, a.y, b);

}

buffer_t<vec3> concat_impl(const buffer_t<vec2>& a, const buffer_t<float>& b){
  const size_t elem = a.elem();
  soil::buffer_t<vec3> output(elem, soil::GPU);
  _concat<<<block(elem, 1024), 1024>>>(output, a, b);
  return output;
}

//
// Index Based Buffer Select
//

template<typename T>
__global__ void _select_index(soil::buffer_t<T> output, const soil::buffer_t<T> source, const soil::buffer_t<int> index_b){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= output.elem()) return;

  const int index = index_b[n];
  output[n] = source[index];

}

buffer select_index_impl(const buffer& buffer, const buffer_t<int>& index){
  return soil::select(buffer.type(), [&]<typename T>() -> soil::buffer {

    const auto buffer_t = buffer.as<T>();
    const size_t elem = index.elem();
    soil::buffer_t<T> output(elem, soil::GPU);

    std::cout<<"impl called converted"<<std::endl;

    _select_index<<<block(elem, 1024), 1024>>>(output, buffer_t, index);
    return soil::buffer(output);

  });
}

} // end of namespace soil

#endif