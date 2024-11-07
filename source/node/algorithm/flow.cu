#pragma diag_suppress 20012

#include <soillib/node/algorithm/flow_test.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <glm/glm.hpp>

template<typename T>
struct gpu_buf{
  T* _data = NULL;
  size_t _size = 0;
};

//template<typename T>
__global__ void _select(gpu_buf<int> in, gpu_buf<glm::ivec2> out){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= in._size) return;

  const int dirmap[8] = {
    7, 8, 1, 2, 3, 4, 5, 6,
  };

  const glm::ivec2 coords[8] = {
    glm::ivec2{-1, 0},
    glm::ivec2{-1, 1},
    glm::ivec2{ 0, 1},
    glm::ivec2{ 1, 1},
    glm::ivec2{ 1, 0},
    glm::ivec2{ 1,-1},
    glm::ivec2{ 0,-1},
    glm::ivec2{-1,-1},
  };

  glm::ivec2 val(0, 0);
  for(size_t k = 0; k < 8; ++k){
    if(in._data[index] == dirmap[k]){
      val = coords[k];
      break;
    }
  }
  out._data[index] = val;

}

void soil::flow_test::test(void* in, const size_t size_in, void* out, const size_t size_out){
  std::cout<<"KERNEL CALLED HERE"<<std::endl;

  // malloc, copy, 
  gpu_buf<int> g_in;
  g_in._size = size_in;
  cudaMalloc(&g_in._data, size_in*sizeof(int));
  cudaMemcpy(g_in._data, in, size_in*sizeof(int), cudaMemcpyHostToDevice);

  gpu_buf<glm::ivec2> g_out;
  g_out._size = size_out;
  cudaMalloc(&g_out._data, size_in*sizeof(glm::ivec2));

  const int thread = 1024;
  const int block = (size_in + thread - 1)/thread;

  _select<<<block, thread>>>(g_in, g_out);

  cudaMemcpy(out, g_out._data, size_out*sizeof(glm::ivec2), cudaMemcpyDeviceToHost);
}