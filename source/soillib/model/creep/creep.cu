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

//! Slope fraction at which the flux law hands over to its linear extension.
#define __CREEP_SMAX 0.99f

//! Volumetric flux per unit width along one link [m^2/time]. Past
//! __CREEP_SMAX the law continues along the tangent taken there, so the flux
//! and its derivative are both continuous.
__device__ inline float __creep_flux (
  const float slope,      //!< Link Slope Magnitude [1]
  const float rate,       //!< Creep Diffusivity    [m^2/time]
  const float critslope   //!< Critical Slope       [1]
) {

  if(critslope <= 0.0f)
    return rate * slope;

  constexpr float smax = __CREEP_SMAX;
  const float s = slope / critslope;

  if(s < smax)
    return rate * slope / (1.0f - s * s);

  const float t = 1.0f - smax * smax;
  const float sm = smax * critslope;
  const float qm = rate * sm / t;
  const float dqm = rate * (1.0f + smax * smax) / (t * t);
  return qm + dqm * (slope - sm);

}

//! Derivative of the link flux, which is what governs stability. Constant on
//! the linear extension, where the slope fraction saturates at __CREEP_SMAX.
__device__ inline float __creep_dflux (
  const float slope,
  const float rate,
  const float critslope
) {

  if(critslope <= 0.0f)
    return rate;

  const float s = fminf(slope / critslope, __CREEP_SMAX);
  const float t = 1.0f - s * s;
  return rate * (1.0f + s * s) / (t * t);

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

  // D8 link set, weighted 2/3 axial and 1/3 diagonal.
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

    // A positive drop runs the link out of this cell: a loss here, a gain for
    // the neighbor. The pair is antisymmetric, so mass conserves without a scatter.
    const float drop = h - hn;
    const float flux = __creep_flux(fabsf(drop) / len, rate, critslope);
    change -= copysignf(weight[k] * flux / len, drop);

  }

  tensorOut[n] = change;

}

__global__ void __creep_relax (
  silt::tensor_t<float> tensorOut,      //!< Output Relaxation Rate [1/time]
  const silt::tensor_t<float> tensorIn, //!< Input Height Field     [1]
  const silt::shape shape,
  const silt::vec3 scale,
  const float rate,
  const float critslope
) {

  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n >= shape.elem()) return;

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

  float lambda = 0.0f;
  for(int k = 0; k < 8; ++k){

    const silt::ivec2 npos = ipos + shift[k];
    if(shape.oob(npos))
      continue;

    const float hn = tensorIn[shape.flatten(npos)] * scale.z;
    const float len = glm::length(silt::vec2(shift[k].x * scale.x, shift[k].y * scale.y));
    lambda += weight[k] * __creep_dflux(fabsf(h - hn) / len, rate, critslope) / (len * len);

  }

  // Differentiated against the normalized height, matching soil::spl_relax, so
  // the same dt serves both the rate and the fit.
  tensorOut[n] = lambda * scale.z;

}

silt::tensor_t<float> creep_relax (
  const silt::tensor_t<float>& height,
  const silt::vec3 scale,
  const float rate,
  const float critslope
) {

  const silt::shape shapeIn = height.shape();
  const silt::shape shape = silt::shape(shapeIn[0], shapeIn[1]);
  auto out = silt::tensor_t<float>(shape, silt::host_t::GPU);
  __creep_relax<<<block(shape.elem(), 512), 512>>>(out, height, shape, scale, rate, critslope);
  gpuErrchk(cudaGetLastError());
  return out;

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
