#ifndef SOILLIB_OP_COMMON_CU
#define SOILLIB_OP_COMMON_CU
#define HAS_CUDA

#include <soillib/op/common.hpp>
#include <soillib/op/gather.hpp>
#include <soillib/util/error.hpp>

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

//
// Resizing Kernels
//

template<typename T>
__global__ void _resize(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs, const flat_t<2> out, const flat_t<2> in){

  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= lhs.elem()){
    return;
  }

  const ivec2 ipos = out.unflatten(index);
  const vec2 fpos = vec2(ipos)/vec2(out[0]-1, out[1]-1);
  const vec2 npos = fpos * vec2(in[0]-1, in[1]-1);

  const unsigned int index_in = in.flatten(npos);
  if(index_in >= rhs.elem()){
    lhs[index] = T(0);
    return;
  }

  const int i00 = in.flatten(npos + vec2(0, 0));
  const int i01 = in.flatten(npos + vec2(0, 1));
  const int i10 = in.flatten(npos + vec2(1, 0));
  const int i11 = in.flatten(npos + vec2(1, 1));

  // Note: This should be part of the lerp type to control for
  if(!in.oob(npos + vec2(1, 1))){
    T v00 = rhs[i00];
    T v01 = rhs[i01];
    T v10 = rhs[i10];
    T v11 = rhs[i11];
    lerp_t lerp(v00, v01, v10, v11, npos - glm::floor(npos));
    lhs[index] = lerp.val();
  } else {
    lhs[index] = rhs[index_in];
  }

}

template<typename T>
void resize_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs, soil::ivec2 out, soil::ivec2 in){
  int thread = 1024;
  int elem = lhs.elem();
  int block = (elem + thread - 1)/thread;

  const flat_t<2> out_t({out.x, out.y});
  const flat_t<2> in_t({in.x, in.y});

  _resize<<<block, thread>>>(lhs, rhs, out_t, in_t);
}

template void resize_impl<int>   (soil::buffer_t<int> lhs,     const soil::buffer_t<int> rhs,     soil::ivec2 out, soil::ivec2 in);
template void resize_impl<float> (soil::buffer_t<float> lhs,   const soil::buffer_t<float> rhs,   soil::ivec2 out, soil::ivec2 in);
template void resize_impl<double>(soil::buffer_t<double> lhs,  const soil::buffer_t<double> rhs,  soil::ivec2 out, soil::ivec2 in);
template void resize_impl<vec2>  (soil::buffer_t<vec2> lhs,    const soil::buffer_t<vec2> rhs,    soil::ivec2 out, soil::ivec2 in);
template void resize_impl<vec3>  (soil::buffer_t<vec3> lhs,    const soil::buffer_t<vec3> rhs,    soil::ivec2 out, soil::ivec2 in);
template void resize_impl<ivec2> (soil::buffer_t<ivec2> lhs,   const soil::buffer_t<ivec2> rhs,   soil::ivec2 out, soil::ivec2 in);
template void resize_impl<ivec3> (soil::buffer_t<ivec3> lhs,   const soil::buffer_t<ivec3> rhs,   soil::ivec2 out, soil::ivec2 in);

} // end of namespace soil

#endif