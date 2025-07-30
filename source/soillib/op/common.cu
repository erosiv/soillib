#ifndef SOILLIB_OP_COMMON_CU
#define SOILLIB_OP_COMMON_CU
#define HAS_CUDA

#include <soillib/op/common.hpp>
#include <soillib/op/gather.hpp>
#include <soillib/core/error.hpp>
#include <soillib/core/operation.hpp>

namespace soil {

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

//
// Unary Operations
//

template<typename T>
void set(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return rhs;
  });
}

template void soil::set<int>   (soil::tensor_t<int> lhs,     const int rhs);
template void soil::set<float> (soil::tensor_t<float> lhs,   const float rhs);
template void soil::set<double>(soil::tensor_t<double> lhs,  const double rhs);

template<typename T>
void add(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a + rhs;
  });
}

template void soil::add<int>   (soil::tensor_t<int> buffer,    const int val);
template void soil::add<float> (soil::tensor_t<float> buffer,  const float val);
template void soil::add<double>(soil::tensor_t<double> buffer, const double val);

template<typename T>
void multiply(tensor_t<T> lhs, const T rhs) {
  op::uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a * rhs;
  });
}

template void soil::multiply<int>   (soil::tensor_t<int> buffer,    const int val);
template void soil::multiply<float> (soil::tensor_t<float> buffer,  const float val);
template void soil::multiply<double>(soil::tensor_t<double> buffer, const double val);

template<typename T>
void clamp(soil::tensor_t<T> lhs, const T min, const T max) {
  op::uniop_inplace(lhs, [min, max] GPU_ENABLE (const T a){
    return glm::clamp(a, min, max);
  });
}

template void soil::clamp<int>   (soil::tensor_t<int> buffer,    const int min, const int max);
template void soil::clamp<float> (soil::tensor_t<float> buffer,  const float min, const float max);
template void soil::clamp<double>(soil::tensor_t<double> buffer, const double min, const double max);

//
// Binary Operations
//

template<typename T>
void set(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return b;
  });
}

template void soil::set<int>   (soil::tensor_t<int> lhs,     const soil::tensor_t<int> rhs);
template void soil::set<float> (soil::tensor_t<float> lhs,   const soil::tensor_t<float> rhs);
template void soil::set<double>(soil::tensor_t<double> lhs,  const soil::tensor_t<double> rhs);

template<typename T>
void add(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a + b;
  });
}

template void soil::add<int>   (soil::tensor_t<int> lhs,     const soil::tensor_t<int> rhs);
template void soil::add<float> (soil::tensor_t<float> lhs,   const soil::tensor_t<float> rhs);
template void soil::add<double>(soil::tensor_t<double> lhs,  const soil::tensor_t<double> rhs);

template<typename T>
void multiply(tensor_t<T> lhs, const tensor_t<T> rhs) {
  op::binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a * b;
  });
}

template void soil::multiply<int>   (soil::tensor_t<int> lhs,     const soil::tensor_t<int> rhs);
template void soil::multiply<float> (soil::tensor_t<float> lhs,   const soil::tensor_t<float> rhs);
template void soil::multiply<double>(soil::tensor_t<double> lhs,  const soil::tensor_t<double> rhs);

template<typename T>
void mix(tensor_t<T> lhs, const tensor_t<T> rhs, const float w) {
  op::binop_inplace(lhs, rhs, [w] GPU_ENABLE (const T a, const T b){
    return (1.0f - w) * a + w * b;
  });
}

template void soil::mix<float> (soil::tensor_t<float> buffer,   const soil::tensor_t<float> rhs, const float w);

//
// Setting Kernels
//

template<typename T>
__global__ void _set(soil::tensor_t<T> lhs, const T val, size_t start, size_t stop, size_t step){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int i = start + n*step;
  if(i >= stop) return;
  lhs[i] = val;
}

template<typename T>
void set_impl(soil::tensor_t<T> lhs, const T val, size_t start, size_t stop, size_t step){
  int thread = 1024;
  int elem = (stop - start + step - 1)/step;
  int block = (elem + thread - 1)/thread;
  _set<<<block, thread>>>(lhs, val, start, stop, step);
}

template void set_impl<int>   (soil::tensor_t<int> buffer,    const int val, size_t start, size_t stop, size_t step);
template void set_impl<float> (soil::tensor_t<float> buffer,  const float val, size_t start, size_t stop, size_t step);
template void set_impl<double>(soil::tensor_t<double> buffer, const double val, size_t start, size_t stop, size_t step);

//
// Other Stuff
//

__global__ void __seed(tensor_t<curandState> buf, const size_t seed, const size_t offset) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buf.elem()) return;
  curand_init(seed, n, offset, &buf[n]);
}

void seed(tensor_t<curandState>& buf, const size_t seed, const size_t offset){
  __seed<<<block(buf.elem(), 512), 512>>>(buf, seed, offset);
  cudaDeviceSynchronize();
}

//
// Resizing Kernels
//

template<typename T>
__global__ void __resize(soil::tensor_t<T> lhs, const soil::tensor_t<T> rhs){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= lhs.elem()){
    return;
  }

  // Normalize Coordinates in Target Frame
  const shape out = lhs.shape();
  const ivec2 ipos = out.unflatten(n);
  const vec2 fpos = vec2(ipos)/vec2(out[0]-1, out[1]-1);
  
  // Unnormalize in Source Frame
  const shape in = rhs.shape();
  const vec2 npos = fpos * vec2(in[0]-1, in[1]-1);
  const int i00 = in.flatten(npos + vec2(0, 0));
  const int i01 = in.flatten(npos + vec2(0, 1));
  const int i10 = in.flatten(npos + vec2(1, 0));
  const int i11 = in.flatten(npos + vec2(1, 1));

  // Linear Interpolation w. Bounds Handling
  if(in.oob(npos)){
    lhs[n] = T(0);
  } else if(in.oob(npos + vec2(1, 1))){
    lhs[n] = rhs[i00]; 
  } else {
    T v00 = rhs[i00];
    T v01 = rhs[i01];
    T v10 = rhs[i10];
    T v11 = rhs[i11];
    lerp_t lerp(v00, v01, v10, v11, npos - glm::floor(npos));
    lhs[n] = lerp.val();
  }

}

template<typename T>
tensor_t<T> resize(const tensor_t<T> rhs, const shape shape){

  if(rhs.host() != soil::host_t::GPU){
    throw soil::error::mismatch_host(soil::host_t::GPU, rhs.host());
  }

  auto lhs = soil::tensor_t<T>(shape, soil::host_t::GPU);
  __resize<<<block(lhs.elem(), 1024), 1024>>>(lhs, rhs);
  return lhs;

}

template soil::tensor_t<int>    soil::resize<int>   (const soil::tensor_t<int> lhs,     const shape shape);
template soil::tensor_t<float>  soil::resize<float> (const soil::tensor_t<float> lhs,   const shape shape);
template soil::tensor_t<double> soil::resize<double>(const soil::tensor_t<double> lhs,  const shape shape);

} // end of namespace soil

#endif