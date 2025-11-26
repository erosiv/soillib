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

template<size_t CHECKERED = 0>
__global__ void __masked_mean (
  silt::tensor_t<float> tensor,      //!< Output Field
  const silt::shape shape,
  const silt::vec2 center,
  const float rad
){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem) return;

  const silt::ivec2 pos = shape.unflatten(n);
  if((pos.x + pos.y)%2 != CHECKERED)
    return;

  if(glm::length(silt::vec2(pos) - center) >= rad)
    return;

  const int in0 = shape.flatten(pos - silt::ivec2(1, 0));
  const int ip0 = shape.flatten(pos + silt::ivec2(1, 0));
  const int i0n = shape.flatten(pos - silt::ivec2(0, 1));
  const int i0p = shape.flatten(pos + silt::ivec2(0, 1));

  const float v00 = tensor[n];
  const float vn0 = shape.oob(pos - silt::ivec2(1, 0)) ? v00 : tensor[in0];
  const float vp0 = shape.oob(pos + silt::ivec2(1, 0)) ? v00 : tensor[ip0];
  const float v0n = shape.oob(pos - silt::ivec2(0, 1)) ? v00 : tensor[i0n];
  const float v0p = shape.oob(pos + silt::ivec2(0, 1)) ? v00 : tensor[i0p];

  tensor[n] = 0.25f * (vn0 + vp0 + v0n + v0p);

}

void masked_mean(silt::tensor_t<float> tensor, const silt::vec2 center, const float rad){
  const silt::shape shape = tensor.shape();
  __masked_mean<0><<<block(shape.elem, 512), 512>>>(tensor, shape, center, rad);
  __masked_mean<1><<<block(shape.elem, 512), 512>>>(tensor, shape, center, rad);
}

} // end of namespace soil

#endif