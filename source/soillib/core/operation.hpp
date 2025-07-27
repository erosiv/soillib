#ifndef SOILLIB_OPERATION
#define SOILLIB_OPERATION

#include <soillib/core/buffer.hpp>
#include <soillib/util/error.hpp>

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

//
// Templated CUDA Implementations
//

#ifdef HAS_CUDA

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

//
// In-Place Binary Operation w. Value
//

template<typename T, typename F>
__global__ void __binop_inplace(buffer_t<T> lhs, const T rhs, F func) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n < lhs.elem()){
    const T a = lhs[n];
    lhs[n] = func(a, rhs);
  }
}

template<typename T, typename F>
void binop_inplace(buffer_t<T> lhs, const T rhs, F func) {

  if(lhs.host() == soil::host_t::GPU){
    __binop_inplace<<<block(lhs.elem(), 512), 512>>>(lhs, rhs, func);
  }

  if(lhs.host() == soil::host_t::CPU){
    for(size_t i = 0; i < lhs.elem(); ++i){
      lhs[i] = func(lhs[i], rhs);
    }
  }

}

//
// Specific Instantiations
//

template<typename T>
void set(buffer_t<T> lhs, const T rhs) {
  binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return b;
  });
}

template<typename T>
void add(buffer_t<T> lhs, const T rhs) {
  binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a + b;
  });
}

template<typename T>
void multiply(buffer_t<T> lhs, const T rhs) {
  binop_inplace(lhs, rhs, [] GPU_ENABLE (const T a, const T b){
    return a * b;
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

  if(lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

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

#endif

}
}

#endif