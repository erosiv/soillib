#ifndef SOILLIB_OPERATION
#define SOILLIB_OPERATION

#include <soillib/core/buffer.hpp>
#include <soillib/util/error.hpp>
#include <curand_kernel.h>

//
// Templated Kernelized Operations on Buffers and Tensors
//
// In essence, we want to de-complexify the current state
// of all the possible operations, which has gotten out of
// hand. To do this, and to make operations available to
// other libraries as well, we will finally make sure that
// this functions correctly with header / source etc.
//

/*
On the one hand, we have to be able to call these operations
CPU side without requiring a cuda compiler. Therefore, we need
to just expose the function calls so that those are available...

But then, on the GPU side, we need to have the direct kernel available.
Which means... ?
*/

namespace soil {
namespace op {

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

//
// Forward Declarations
//

template<typename T>
void set(buffer_t<T> lhs, const T value);

template<typename T>
void add(buffer_t<T> lhs, const T value);

template<typename T>
void multiply(buffer_t<T> lhs, const T value);

template<typename T>
void set(buffer_t<T> lhs, const buffer_t<T> rhs);

template<typename T>
void add(buffer_t<T> lhs, const buffer_t<T> rhs);

template<typename T>
void multiply(buffer_t<T> lhs, const buffer_t<T> rhs);

template<typename T>
void mix(buffer_t<T> lhs, const buffer_t<T> rhs, const float w);

void seed(buffer_t<curandState>& buf, const size_t seed, const size_t offset);

//
// Templated CUDA Implementations
//

#ifdef HAS_CUDA

//
// In-Place Binary Operation w. Value
//

template<typename T, typename F>
__global__ void __uniop_inplace(buffer_t<T> lhs, F f){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n < lhs.elem()){
    lhs[n] = f(lhs[n]);
  }
}

template<typename T, typename F>
void uniop_inplace(buffer_t<T> lhs, F func) {

  if(lhs.host() == soil::host_t::GPU){
    __uniop_inplace<<<block(lhs.elem(), 512), 512>>>(lhs, func);
  }

  if(lhs.host() == soil::host_t::CPU){
    for(size_t i = 0; i < lhs.elem(); ++i){
      lhs[i] = func(lhs[i]);
    }
  }

}

//
// Specific Instantiations
//

template<typename T>
void set(buffer_t<T> lhs, const T rhs) {
  uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return rhs;
  });
}

template<typename T>
void add(buffer_t<T> lhs, const T rhs) {
  uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a + rhs;
  });
}

template<typename T>
void multiply(buffer_t<T> lhs, const T rhs) {
  uniop_inplace(lhs, [rhs] GPU_ENABLE (const T a){
    return a * rhs;
  });
}

//
// In-Place Binary Operation w. Buffer
//

template<typename T, typename F>
__global__ void __binop_inplace(buffer_t<T> lhs, const buffer_t<T> rhs, F func) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n < lhs.elem()){
    const T a = lhs[n];
    const T b = rhs[n];
    lhs[n] = func(a, b);
  }
}

template<typename T, typename F>
void binop_inplace(buffer_t<T> lhs, const buffer_t<T> rhs, F func) {

  if(lhs.host() == soil::host_t::GPU){
    __binop_inplace<<<block(lhs.elem(), 512), 512>>>(lhs, rhs, func);
  }

  if(lhs.host() == soil::host_t::CPU){
    for(size_t i = 0; i < lhs.elem(); ++i){
      lhs[i] = func(lhs[i], rhs[i]);
    }
  }

}

//
// Specific Instantiations
//

template<typename T>
void set(buffer_t<T> lhs, const buffer_t<T> rhs) {
  binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return b;
  });
}

template<typename T>
void add(buffer_t<T> lhs, const buffer_t<T> rhs) {
  binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a + b;
  });
}

template<typename T>
void multiply(buffer_t<T> lhs, const buffer_t<T> rhs) {
  binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a * b;
  });
}

template<typename T>
void mix(buffer_t<T> lhs, const buffer_t<T> rhs, const float w) {
  binop_inplace(lhs, rhs, [w] GPU_ENABLE (const T a, const T b){
    return (1.0f - w) * a + w * b;
  });
}

#endif

//
// Legacy Functions
//! \todo get rid of this...

template<typename To, typename From>
void copy(soil::buffer_t<To> &out, const soil::buffer_t<From> &in, vec2 gmin, vec2 gmax, vec2 gscale, vec2 wmin, vec2 wmax, vec2 wscale, float pscale) {

  const ivec2 pmin = ivec2(pscale * (gmin - wmin) / wscale);
  const ivec2 pmax = ivec2(pscale * (gmax - wmin) / wscale);
  const ivec2 pext = ivec2(pscale * (wmax - wmin) / wscale);
  const ivec2 gext = ivec2((gmax - gmin) / gscale);

  for (int x = pmin[1]; x < pmax[1]; ++x) {
    for (int y = pmin[0]; y < pmax[0]; ++y) {

      const int ind_out = y + pext[0] * (pext[1] - x - 1);

      const size_t px = size_t((pmax[1] - x - 1) / pscale);
      const size_t py = size_t((y - pmin[0]) / pscale);
      const size_t ind_in = py + px * gext[0];

      out[ind_out] = To(From(pscale) * in[ind_in]);
    }
  }
}

}
}

#endif