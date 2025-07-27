#ifndef SOILLIB_OP_COMMON_CU_HEADER
#define SOILLIB_OP_COMMON_CU_HEADER
#define HAS_CUDA

// CUDA Kernel Header Implementations
//  The idea is that we can write kernel
//  code that is compiled in multiple
//  units at the same time, collected commonly
//  here... sometimes you need device functions
//  to exist in multiple places at once.

#include <soillib/core/buffer.hpp>
#include <curand_kernel.h>

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

namespace soil {
namespace {

//
// Seed Random Data
//

__global__ void __seed(buffer_t<curandState> buf, const size_t seed, const size_t offset) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buf.elem()) return;
  curand_init(seed, n, offset, &buf[n]);
}

void seed(buffer_t<curandState>& buf, const size_t seed, const size_t offset) {
  __seed<<<block(buf.elem(), 512), 512>>>(buf, seed, offset);
  cudaDeviceSynchronize();
}

//
// Set Buffer Type
//

template<typename T>
__global__ void __set(buffer_t<T> buf, const T val){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buf.elem()) return;
  buf[n] = val;
}

template<typename T>
void set(buffer_t<T>& buf, const T val){
  __set<<<block(buf.elem(), 512), 512>>>(buf, val);
}

//
// Exponential Filter Operation
//

template<typename T>
__global__ void __filter(buffer_t<T> buf_A, const buffer_t<T> buf_B, const float w){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buf_A.elem()) return;
  if(n >= buf_B.elem()) return;
  buf_A[n] = glm::mix(buf_A[n], buf_B[n], w);
}

template<typename T>
void filter(buffer_t<T>& buf_A, const buffer_t<T>& buf_B, const float w){
  __filter<<<block(buf_A.elem(), 512), 512>>>(buf_A, buf_B, w);
}

}
} // end of namespace soil

#endif