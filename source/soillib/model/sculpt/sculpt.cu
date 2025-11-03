#ifndef SOILLIB_MODEL_SCULPT_CU
#define SOILLIB_MODEL_SCULPT_CU
#define HAS_CUDA

#include <soillib/model/sculpt/sculpt.hpp>

namespace soil {

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

__global__ void __masked_set(
  silt::tensor_t<float> tensor,      //!< Output Field
  const silt::shape shape,
  const float value,
  const silt::vec2 center,
  const float rad
){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem) return;

  const silt::vec2 pos = shape.unflatten(n);
  if(glm::length(pos - center) < rad){
    tensor[n] = value;
  }
}

void masked_set(silt::tensor_t<float> tensor, const float value, const silt::vec2 center, const float rad){
  const silt::shape shape = tensor.shape();
  __masked_set<<<block(shape.elem, 512), 512>>>(tensor, shape, value, center, rad);
}

} // end of namespace soil

#endif