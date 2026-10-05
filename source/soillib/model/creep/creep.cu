#include <soillib/soillib.hpp>

#include <soillib/model/creep/creep.hpp>

namespace soil {

namespace {

inline int block(const int64_t elem, const int thread) {
  return int((elem + thread - 1) / thread);
}

}

//
// Hillslope Creep: Divergence of a Nonlinear Downhill Flux
//

//! Volumetric flux per unit width along one link [m^2/time].
//! The denominator is floored so the diffusivity stays bounded.
__device__ inline float __creep_flux (
  const float slope,      //!< Link Slope Magnitude [1]
  const float rate,       //!< Creep Diffusivity    [m^2/time]
  const float critslope   //!< Critical Slope       [1]
) {

  if(critslope <= 0.0f)
    return rate * slope;

  constexpr float smax = 0.99f;
  const float s = fminf(slope / critslope, smax);
  return rate * slope / (1.0f - s * s);

}

__global__ void __creep (
  silt::tensor_t<float> tensorOut,      //!< Output Change Rate Field [m/time]
  const silt::tensor_t<float> tensorIn, //!< Input Height Field       [1]
  const silt::shape shape,
  const silt::vec3 scale,
  const float rate,
  const float critslope
) {

  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n >= shape.elem()) return;

  // D8 link set. Axial links carry 2/3 and diagonal links 1/3 so that a linear
  // flux law collapses onto the isotropic nine-point Laplacian.
  const silt::ivec2 shift[8] = {
    {-1, 0}, { 1, 0}, { 0,-1}, { 0, 1},
    {-1,-1}, { 1, 1}, { 1,-1}, {-1, 1},
  };
  const float weight[8] = {
    2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f,
    1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f,
  };

  const silt::ivec2 ipos = shape.unflatten(n);
  const float h = tensorIn[n] * scale.z;

  float change = 0.0f;
  for(int k = 0; k < 8; ++k){

    const silt::ivec2 npos = ipos + shift[k];
    if(shape.oob(npos))    //!< Closed boundary: a link out of the domain is inert.
      continue;

    const float hn = tensorIn[shape.flatten(npos)] * scale.z;
    const float len = glm::length(silt::vec2(shift[k].x * scale.x, shift[k].y * scale.y));

    // A positive drop means the link runs out of this cell, so its flux is a
    // loss here and a gain for the neighbor. The pair is antisymmetric, which
    // is what conserves mass without a scatter.
    const float drop = h - hn;
    const float flux = __creep_flux(fabsf(drop) / len, rate, critslope);
    change -= copysignf(weight[k] * flux / len, drop);

  }

  tensorOut[n] = change;

}

silt::tensor_t<float> creep (
  const silt::tensor_t<float>& height,
  const silt::vec3 scale,
  const float rate,
  const float critslope
) {

  const silt::shape shapeIn = height.shape();
  const silt::shape shape = silt::shape(shapeIn[0], shapeIn[1]);
  auto out = silt::tensor_t<float>(shape, silt::host_t::GPU);
  __creep<<<block(shape.elem(), 512), 512>>>(out, height, shape, scale, rate, critslope);
  gpuErrchk(cudaGetLastError());
  return out;

}

} // end of namespace soil
