#ifndef SOILLIB_OP_COMMON_CU
#define SOILLIB_OP_COMMON_CU
#define HAS_CUDA

#include <soillib/op/common.hpp>
#include <soillib/op/gather.hpp>
#include <soillib/util/error.hpp>
#include <soillib/core/operation.hpp>

namespace soil {

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

//
// Specific Instantiations
//

// Unary Operations

template<typename T>
void set(buffer_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return rhs;
  });
}

template<typename T>
void add(buffer_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a + rhs;
  });
}

template<typename T>
void multiply(buffer_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a * rhs;
  });
}

template<typename T>
void clamp(soil::buffer_t<T> lhs, const T min, const T max) {
  op::uniop_inplace(lhs, [min, max] GPU_ENABLE (const T a){
    return glm::clamp(a, min, max);
  });
}

// Binary Operations

template<typename T>
void set(buffer_t<T> lhs, const buffer_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return b;
  });
}

template<typename T>
void add(buffer_t<T> lhs, const buffer_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a + b;
  });
}

template<typename T>
void multiply(buffer_t<T> lhs, const buffer_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a * b;
  });
}

template<typename T>
void mix(buffer_t<T> lhs, const buffer_t<T> rhs, const float w) {
  op::binop_inplace(lhs, rhs, [w] GPU_ENABLE (const T a, const T b){
    return (1.0f - w) * a + w * b;
  });
}

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

// Explicit Template Instantiations
template void soil::set<int>   (soil::buffer_t<int> lhs,     const soil::buffer_t<int> rhs);
template void soil::set<float> (soil::buffer_t<float> lhs,   const soil::buffer_t<float> rhs);
template void soil::set<double>(soil::buffer_t<double> lhs,  const soil::buffer_t<double> rhs);
template void soil::set<vec2>  (soil::buffer_t<vec2> lhs,    const soil::buffer_t<vec2> rhs);
template void soil::set<vec3>  (soil::buffer_t<vec3> lhs,    const soil::buffer_t<vec3> rhs);
template void soil::set<ivec2> (soil::buffer_t<ivec2> lhs,   const soil::buffer_t<ivec2> rhs);
template void soil::set<ivec3> (soil::buffer_t<ivec3> lhs,   const soil::buffer_t<ivec3> rhs);

template void soil::set<int>   (soil::buffer_t<int> lhs,     const int rhs);
template void soil::set<float> (soil::buffer_t<float> lhs,   const float rhs);
template void soil::set<double>(soil::buffer_t<double> lhs,  const double rhs);
template void soil::set<vec2>  (soil::buffer_t<vec2> lhs,    const vec2 rhs);
template void soil::set<vec3>  (soil::buffer_t<vec3> lhs,    const vec3 rhs);
template void soil::set<ivec2> (soil::buffer_t<ivec2> lhs,   const ivec2 rhs);
template void soil::set<ivec3> (soil::buffer_t<ivec3> lhs,   const ivec3 rhs);

// Explicit Template Instantiations
template void soil::add<int>   (soil::buffer_t<int> lhs,     const soil::buffer_t<int> rhs);
template void soil::add<float> (soil::buffer_t<float> lhs,   const soil::buffer_t<float> rhs);
template void soil::add<double>(soil::buffer_t<double> lhs,  const soil::buffer_t<double> rhs);
template void soil::add<vec2>  (soil::buffer_t<vec2> lhs,    const soil::buffer_t<vec2> rhs);
template void soil::add<vec3>  (soil::buffer_t<vec3> lhs,    const soil::buffer_t<vec3> rhs);
template void soil::add<ivec2> (soil::buffer_t<ivec2> lhs,   const soil::buffer_t<ivec2> rhs);
template void soil::add<ivec3> (soil::buffer_t<ivec3> lhs,   const soil::buffer_t<ivec3> rhs);

template void soil::add<int>   (soil::buffer_t<int> buffer,    const int val);
template void soil::add<float> (soil::buffer_t<float> buffer,  const float val);
template void soil::add<double>(soil::buffer_t<double> buffer, const double val);
template void soil::add<vec2>  (soil::buffer_t<vec2> buffer,   const vec2 val);
template void soil::add<vec3>  (soil::buffer_t<vec3> buffer,   const vec3 val);
template void soil::add<ivec2> (soil::buffer_t<ivec2> buffer,  const ivec2 val);
template void soil::add<ivec3> (soil::buffer_t<ivec3> buffer,  const ivec3 val);

template void soil::multiply<int>   (soil::buffer_t<int> buffer,    const int val);
template void soil::multiply<float> (soil::buffer_t<float> buffer,  const float val);
template void soil::multiply<double>(soil::buffer_t<double> buffer, const double val);
template void soil::multiply<vec2>  (soil::buffer_t<vec2> buffer,   const vec2 val);
template void soil::multiply<vec3>  (soil::buffer_t<vec3> buffer,   const vec3 val);
template void soil::multiply<ivec2> (soil::buffer_t<ivec2> buffer,  const ivec2 val);
template void soil::multiply<ivec3> (soil::buffer_t<ivec3> buffer,  const ivec3 val);

template void soil::multiply<int>   (soil::buffer_t<int> lhs,     const soil::buffer_t<int> rhs);
template void soil::multiply<float> (soil::buffer_t<float> lhs,   const soil::buffer_t<float> rhs);
template void soil::multiply<double>(soil::buffer_t<double> lhs,  const soil::buffer_t<double> rhs);
template void soil::multiply<vec2>  (soil::buffer_t<vec2> lhs,    const soil::buffer_t<vec2> rhs);
template void soil::multiply<vec3>  (soil::buffer_t<vec3> lhs,    const soil::buffer_t<vec3> rhs);
template void soil::multiply<ivec2> (soil::buffer_t<ivec2> lhs,   const soil::buffer_t<ivec2> rhs);
template void soil::multiply<ivec3> (soil::buffer_t<ivec3> lhs,   const soil::buffer_t<ivec3> rhs);

template void soil::mix<float> (soil::buffer_t<float> buffer,   const soil::buffer_t<float> rhs, const float w);
template void soil::mix<vec2> (soil::buffer_t<vec2> buffer,     const soil::buffer_t<vec2> rhs, const float w);

// template void op::clamp<int>   (soil::buffer_t<int> buffer,    const int min, const int max);
template void soil::clamp<float> (soil::buffer_t<float> buffer,  const float min, const float max);
//template void op::clamp<double>(soil::buffer_t<double> buffer, const double min, const double max);
//template void op::clamp<vec2>  (soil::buffer_t<vec2> buffer,   const vec2 min, const vec2 max);
//template void op::clamp<vec3>  (soil::buffer_t<vec3> buffer,   const vec3 min, const vec3 max);
//template void op::clamp<ivec2> (soil::buffer_t<ivec2> buffer,  const ivec2 min, const ivec2 max);
//template void op::clamp<ivec3> (soil::buffer_t<ivec3> buffer,  const ivec3 min, const ivec3 max);

__global__ void __seed(buffer_t<curandState> buf, const size_t seed, const size_t offset) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buf.elem()) return;
  curand_init(seed, n, offset, &buf[n]);
}

void seed(buffer_t<curandState>& buf, const size_t seed, const size_t offset){
  __seed<<<block(buf.elem(), 512), 512>>>(buf, seed, offset);
  cudaDeviceSynchronize();
}

//
// Resizing Kernels
//

template<typename T>
__global__ void _resize(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs, const soil::shape out, const soil::shape in){

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

  const soil::shape out_t(out.x, out.y);
  const soil::shape in_t(in.x, in.y);

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