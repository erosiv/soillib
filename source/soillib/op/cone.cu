#include <soillib/soillib.hpp>

#include <soillib/op/cone.hpp>

namespace soil {

namespace {

inline int block(const int64_t elem, const int thread) {
  return int((elem + thread - 1) / thread);
}

}

__global__ void __cone (
  silt::tensor_t<float> tensorOut,  //!< Output Field
  const silt::shape shape,
  const silt::vec2 apex,
  const float height,
  const float dUdx,
  const float dUdy,
  const silt::vec2 scale
){

  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n >= shape.elem()) return;

  const silt::ivec2 ipos = shape.unflatten(n);
  const float dx = (float(ipos.x) - apex.x) * scale.x;
  const float dy = (float(ipos.y) - apex.y) * scale.y;

  tensorOut[n] = height - fabsf(dUdx * dx) - fabsf(dUdy * dy);

}

silt::tensor_t<float> cone (
  const silt::shape shape,
  const silt::vec2 apex,
  const float height,
  const float dUdx,
  const float dUdy,
  const silt::vec2 scale
) {

  auto output = silt::tensor_t<float>(shape, silt::host_t::GPU);
  __cone<<<block(shape.elem(), 512), 512>>>(output, shape, apex, height, dUdx, dUdy, scale);
  gpuErrchk(cudaGetLastError());
  return output;

}

} // end of namespace soil
