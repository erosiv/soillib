#include <soillib/soillib.hpp>

#include <soillib/op/normal.hpp>
#include <soillib/model/grad/grad.hpp>

namespace soil {

namespace {

inline int block(const int64_t elem, const int thread) {
  return int((elem + thread - 1) / thread);
}

}

//
// Gradient to Normal Transform
//

__global__ void __normal (
  silt::tensor_t<float> tensorOut,      //!< Output Normal Field
  const silt::tensor_t<float> tensorIn, //!< Input Gradient Field (dz/dx, dz/dy)
  const silt::shape shape
){

  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n >= shape.elem()) return;

  const auto gradView = tensorIn.view<silt::vec2>();
  auto normalView = tensorOut.view<silt::vec3>();

  const silt::vec2 g = gradView[n];
  normalView[n] = glm::normalize(silt::vec3(-g.x, -g.y, 1.0f));

}

silt::tensor_t<float> normal(const silt::tensor_t<float>& tensor, const silt::vec3 scale) {

  const silt::shape shape_in = tensor.shape();
  if (shape_in.dim() != 2)
    throw std::invalid_argument("normal map can not be computed for non 2D-indexed buffers");

  // Fold the vertical scale into the horizontal spacing, so that
  // gradient() (which expects world-space spacing) yields dz/dx, dz/dy.
  const silt::vec2 gscale(scale.x / scale.z, scale.y / scale.z);
  const silt::tensor_t<float> grad = soil::gradient(tensor, gscale);

  const silt::shape shape_out = silt::shape(shape_in[0], shape_in[1], 3);
  auto output = silt::tensor_t<float>(shape_out, silt::host_t::GPU);

  __normal<<<block(shape_in.elem(), 512), 512>>>(output, grad, shape_in);
  gpuErrchk(cudaGetLastError());

  return output;

}

} // end of namespace soil
