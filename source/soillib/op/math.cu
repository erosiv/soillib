#ifndef SOILLIB_NODE_COMMON
#define SOILLIB_NODE_COMMON
#define HAS_CUDA

#include <soillib/soillib.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <soillib/op/common.hpp>

namespace soil {

//
// Resample Kernels
//

template<typename T, typename Index, typename Flat>
__global__ void _resample(soil::buffer_t<T> input, soil::buffer_t<T> output, const Index index, const Flat flat){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= flat.elem()) return;

  auto pos = flat.unflatten(n);
  if(!index.oob(pos)){
    output[n] = input[index.flatten(pos)];
  }
}

template<typename T>
soil::buffer_t<T> resample_impl(const soil::buffer_t<T>& input, const soil::index& index){

  return select(index.type(), [&]<typename I>(){

    auto index_t = index.as<I>();
    soil::flat_t<I::n_dims> flat(index_t.ext());

    soil::buffer_t<T> output(flat.elem(), soil::GPU);
    using V = soil::typedesc<T>::value_t;
    T value = T{std::numeric_limits<V>::quiet_NaN()};
    set_impl<T>(output, value, 0, flat.elem(), 1);

    int thread = 1024;
    int elem = flat.elem();
    int block = (elem + thread - 1)/thread;
    _resample<<<block, thread>>>(input, output, index_t, flat);

    return output;

  });

}

template soil::buffer_t<int>    resample_impl<int>   (const soil::buffer_t<int>& buffer,    const soil::index& index);
template soil::buffer_t<float>  resample_impl<float> (const soil::buffer_t<float>& buffer,  const soil::index& index);
template soil::buffer_t<double> resample_impl<double>(const soil::buffer_t<double>& buffer, const soil::index& index);
template soil::buffer_t<vec2>   resample_impl<vec2>  (const soil::buffer_t<vec2>& buffer,   const soil::index& index);
template soil::buffer_t<vec3>   resample_impl<vec3>  (const soil::buffer_t<vec3>& buffer,   const soil::index& index);
template soil::buffer_t<ivec2>  resample_impl<ivec2> (const soil::buffer_t<ivec2>& buffer,  const soil::index& index);
template soil::buffer_t<ivec3>  resample_impl<ivec3> (const soil::buffer_t<ivec3>& buffer,  const soil::index& index);

//
// Addition Kernels
//

template<typename T>
__global__ void _add(soil::buffer_t<T> buf, const T val){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < buf.elem())
    buf[index] += val;
}

template<typename T>
__global__ void _add(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < lhs.elem())
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
  if(index < buf.elem())
    buf[index] *= val;
}

template<typename T>
__global__ void _multiply(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < lhs.elem())
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