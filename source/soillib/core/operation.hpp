#ifndef SOILLIB_OPERATION
#define SOILLIB_OPERATION

//
// Generic Template Operations for Buffers and Tensors
//  Written to function both on the GPU and the CPU.
//

#include <soillib/core/buffer.hpp>
#include <soillib/util/error.hpp>
#include <curand_kernel.h>

namespace soil {
namespace op {

//
// Templated CUDA Implementations
//

#ifdef HAS_CUDA

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

// In-Place Operation Kernels

template<typename T, typename F>
__global__ void __uniop_inplace(buffer_t<T> lhs, F f){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n < lhs.elem()){
    lhs[n] = f(lhs[n]);
  }
}

template<typename T, typename F>
__global__ void __binop_inplace(buffer_t<T> lhs, const buffer_t<T> rhs, F func) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n < lhs.elem()){
    const T a = lhs[n];
    const T b = rhs[n];
    lhs[n] = func(a, b);
  }
}

// In-Place Operation Host Functions

template<typename T, typename F>
void uniop_inplace(buffer_t<T> lhs, F func) {

  if(lhs.host() == soil::host_t::CPU){
    for(size_t i = 0; i < lhs.elem(); ++i){
      lhs[i] = func(lhs[i]);
    }
  }

  else if(lhs.host() == soil::host_t::GPU){
    __uniop_inplace<<<block(lhs.elem(), 512), 512>>>(lhs, func);
  }

}

template<typename T, typename F>
void binop_inplace(buffer_t<T> lhs, const buffer_t<T> rhs, F func) {

  if(lhs.host() == soil::host_t::CPU){
    for(size_t i = 0; i < lhs.elem(); ++i){
      lhs[i] = func(lhs[i], rhs[i]);
    }
  }

  else if(lhs.host() == soil::host_t::GPU){
    __binop_inplace<<<block(lhs.elem(), 512), 512>>>(lhs, rhs, func);
  }

}

#endif

}
}

#endif