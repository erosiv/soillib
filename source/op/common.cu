#ifndef SOILLIB_OP_COMMON_CU
#define SOILLIB_OP_COMMON_CU
#define HAS_CUDA

#include <soillib/op/common.hpp>

namespace soil {

//
// Setting Kernels
//

template<typename T>
__global__ void _set(soil::buffer_t<T> buf, const T val, size_t start, size_t stop, size_t step){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int i = start + n*step;
  if(i >= stop) return;
  buf[i] = val;
}

template<typename T>
void set_impl(soil::buffer_t<T> buf, const T val, size_t start, size_t stop, size_t step){
  int thread = 1024;
  int elem = (stop - start + step - 1)/step;
  int block = (elem + thread - 1)/thread;
  _set<<<block, thread>>>(buf, val, start, stop, step);
}

template void set_impl<int>   (soil::buffer_t<int> buffer,    const int val, size_t start, size_t stop, size_t step);
template void set_impl<float> (soil::buffer_t<float> buffer,  const float val, size_t start, size_t stop, size_t step);
template void set_impl<double>(soil::buffer_t<double> buffer, const double val, size_t start, size_t stop, size_t step);
template void set_impl<vec2>  (soil::buffer_t<vec2> buffer,   const vec2 val, size_t start, size_t stop, size_t step);
template void set_impl<vec3>  (soil::buffer_t<vec3> buffer,   const vec3 val, size_t start, size_t stop, size_t step);
template void set_impl<ivec2> (soil::buffer_t<ivec2> buffer,  const ivec2 val, size_t start, size_t stop, size_t step);
template void set_impl<ivec3> (soil::buffer_t<ivec3> buffer,  const ivec3 val, size_t start, size_t stop, size_t step);

template<typename T>
__global__ void _set(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < lhs.elem())
    lhs[index] = rhs[index];
}

template<typename T>
void set_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs){
  int thread = 1024;
  int elem = lhs.elem();
  int block = (elem + thread - 1)/thread;
  _set<<<block, thread>>>(lhs, rhs);
}

template void set_impl<int>   (soil::buffer_t<int> lhs,     const soil::buffer_t<int> rhs);
template void set_impl<float> (soil::buffer_t<float> lhs,   const soil::buffer_t<float> rhs);
template void set_impl<double>(soil::buffer_t<double> lhs,  const soil::buffer_t<double> rhs);
template void set_impl<vec2>  (soil::buffer_t<vec2> lhs,    const soil::buffer_t<vec2> rhs);
template void set_impl<vec3>  (soil::buffer_t<vec3> lhs,    const soil::buffer_t<vec3> rhs);
template void set_impl<ivec2> (soil::buffer_t<ivec2> lhs,   const soil::buffer_t<ivec2> rhs);
template void set_impl<ivec3> (soil::buffer_t<ivec3> lhs,   const soil::buffer_t<ivec3> rhs);

} // end of namespace soil

#endif