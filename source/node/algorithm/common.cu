#ifndef SOILLIB_NODE_ALGORITHM_COMMON
#define SOILLIB_NODE_ALGORITHM_COMMON

#define HAS_CUDA
#include <soillib/soillib.hpp>
#include <soillib/core/buffer.hpp>

namespace soil {

//
// Setting Kernels
//

template<typename T>
__global__ void _set(soil::buffer_t<T> buf, const T val){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < buf.size())
    buf[index] = val;
}

template<typename T>
__global__ void _set(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < lhs.size())
    lhs[index] = rhs[index];
}

template<typename T>
void set_impl(soil::buffer_t<T> buf, const T val){
  int thread = 1024;
  int elem = buf.elem();
  int block = (elem + thread - 1)/thread;
  _set<<<block, thread>>>(buf, val);
}

template<typename T>
void set_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs){
  int thread = 1024;
  int elem = lhs.elem();
  int block = (elem + thread - 1)/thread;
  _set<<<block, thread>>>(lhs, rhs);
}

template void set_impl<int>   (soil::buffer_t<int> buffer,    const int val);
template void set_impl<float> (soil::buffer_t<float> buffer,  const float val);
template void set_impl<double>(soil::buffer_t<double> buffer, const double val);
template void set_impl<vec2>  (soil::buffer_t<vec2> buffer,   const vec2 val);
template void set_impl<vec3>  (soil::buffer_t<vec3> buffer,   const vec3 val);
template void set_impl<ivec2> (soil::buffer_t<ivec2> buffer,  const ivec2 val);
template void set_impl<ivec3> (soil::buffer_t<ivec3> buffer,  const ivec3 val);

template void set_impl<int>   (soil::buffer_t<int> lhs,     const soil::buffer_t<int> rhs);
template void set_impl<float> (soil::buffer_t<float> lhs,   const soil::buffer_t<float> rhs);
template void set_impl<double>(soil::buffer_t<double> lhs,  const soil::buffer_t<double> rhs);
template void set_impl<vec2>  (soil::buffer_t<vec2> lhs,    const soil::buffer_t<vec2> rhs);
template void set_impl<vec3>  (soil::buffer_t<vec3> lhs,    const soil::buffer_t<vec3> rhs);
template void set_impl<ivec2> (soil::buffer_t<ivec2> lhs,   const soil::buffer_t<ivec2> rhs);
template void set_impl<ivec3> (soil::buffer_t<ivec3> lhs,   const soil::buffer_t<ivec3> rhs);

//
// Addition Kernels
//

template<typename T>
__global__ void _add(soil::buffer_t<T> buf, const T val){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < buf.size())
    buf[index] += val;
}

template<typename T>
__global__ void _add(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < lhs.size())
    lhs[index] += rhs[index];
}

template<typename T>
void add_impl(soil::buffer_t<T> buf, const T val){
  int thread = 1024;
  int elem = buf.elem();
  int block = (elem + thread - 1)/thread;
  _add<<<block, thread>>>(buf, val);
}

template<typename T>
void add_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs){
  int thread = 1024;
  int elem = lhs.elem();
  int block = (elem + thread - 1)/thread;
  _add<<<block, thread>>>(lhs, rhs);
}

template void add_impl<int>   (soil::buffer_t<int> buffer,    const int val);
template void add_impl<float> (soil::buffer_t<float> buffer,  const float val);
template void add_impl<double>(soil::buffer_t<double> buffer, const double val);
template void add_impl<vec2>  (soil::buffer_t<vec2> buffer,   const vec2 val);
template void add_impl<vec3>  (soil::buffer_t<vec3> buffer,   const vec3 val);
template void add_impl<ivec2> (soil::buffer_t<ivec2> buffer,  const ivec2 val);
template void add_impl<ivec3> (soil::buffer_t<ivec3> buffer,  const ivec3 val);

template void add_impl<int>   (soil::buffer_t<int> lhs,     const soil::buffer_t<int> rhs);
template void add_impl<float> (soil::buffer_t<float> lhs,   const soil::buffer_t<float> rhs);
template void add_impl<double>(soil::buffer_t<double> lhs,  const soil::buffer_t<double> rhs);
template void add_impl<vec2>  (soil::buffer_t<vec2> lhs,    const soil::buffer_t<vec2> rhs);
template void add_impl<vec3>  (soil::buffer_t<vec3> lhs,    const soil::buffer_t<vec3> rhs);
template void add_impl<ivec2> (soil::buffer_t<ivec2> lhs,   const soil::buffer_t<ivec2> rhs);
template void add_impl<ivec3> (soil::buffer_t<ivec3> lhs,   const soil::buffer_t<ivec3> rhs);

//
// Multiplication Kernels
//

template<typename T>
__global__ void _multiply(soil::buffer_t<T> buf, const T val){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < buf.size())
    buf[index] *= val;
}

template<typename T>
__global__ void _multiply(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < lhs.size())
    lhs[index] *= rhs[index];
}

template<typename T>
void multiply_impl(soil::buffer_t<T> buf, const T val){
  int thread = 1024;
  int elem = buf.elem();
  int block = (elem + thread - 1)/thread;
  _multiply<<<block, thread>>>(buf, val);
}

template<typename T>
void multiply_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs){
  int thread = 1024;
  int elem = lhs.elem();
  int block = (elem + thread - 1)/thread;
  _multiply<<<block, thread>>>(lhs, rhs);
}

template void multiply_impl<int>   (soil::buffer_t<int> buffer,    const int val);
template void multiply_impl<float> (soil::buffer_t<float> buffer,  const float val);
template void multiply_impl<double>(soil::buffer_t<double> buffer, const double val);
template void multiply_impl<vec2>  (soil::buffer_t<vec2> buffer,   const vec2 val);
template void multiply_impl<vec3>  (soil::buffer_t<vec3> buffer,   const vec3 val);
template void multiply_impl<ivec2> (soil::buffer_t<ivec2> buffer,  const ivec2 val);
template void multiply_impl<ivec3> (soil::buffer_t<ivec3> buffer,  const ivec3 val);

template void multiply_impl<int>   (soil::buffer_t<int> lhs,     const soil::buffer_t<int> rhs);
template void multiply_impl<float> (soil::buffer_t<float> lhs,   const soil::buffer_t<float> rhs);
template void multiply_impl<double>(soil::buffer_t<double> lhs,  const soil::buffer_t<double> rhs);
template void multiply_impl<vec2>  (soil::buffer_t<vec2> lhs,    const soil::buffer_t<vec2> rhs);
template void multiply_impl<vec3>  (soil::buffer_t<vec3> lhs,    const soil::buffer_t<vec3> rhs);
template void multiply_impl<ivec2> (soil::buffer_t<ivec2> lhs,   const soil::buffer_t<ivec2> rhs);
template void multiply_impl<ivec3> (soil::buffer_t<ivec3> lhs,   const soil::buffer_t<ivec3> rhs);

/*
//
// Inversion Kernels
//

template<typename T>
__global__ void _invert(soil::buffer_t<T> buf){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < buf.size())
    buf[index] = 1.0 / buf[index];
}

template<typename T>
void invert_impl(soil::buffer_t<T> buf){
  int thread = 1024;
  int elem = buf.elem();
  int block = (elem + thread - 1)/thread;
  _invert<<<block, thread>>>(buf);
}

template void invert_impl<int>   (soil::buffer_t<int> buffer);
template void invert_impl<float> (soil::buffer_t<float> buffer);
template void invert_impl<double>(soil::buffer_t<double> buffer);
template void invert_impl<vec2>  (soil::buffer_t<vec2> buffer);
template void invert_impl<vec3>  (soil::buffer_t<vec3> buffer);
template void invert_impl<ivec2> (soil::buffer_t<ivec2> buffer);
template void invert_impl<ivec3> (soil::buffer_t<ivec3> buffer);
*/

}

#endif