#ifndef SOILLIB_MODEL_PATH_CU
#define SOILLIB_MODEL_PATH_CU
#define HAS_CUDA

#include <soillib/model/path/path.hpp>
#include <silt/core/operation.hpp>
#include <silt/op/common.hpp>
#include <math_constants.h>
#include "sample.hpp"

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

template<size_t D>
__global__ void __solve_uniform (
  silt::tensor_t<float> flux,           //!< Flux Integral Estimate [X/s]
  const silt::tensor_t<float> flow,     //!< Flow-Field Tensor      [m/s]
  const silt::tensor_t<float> source,   //!< Source Term Tensor     [X/s]
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor      [1/s]
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source   []
  const silt::shape shape,              //!< Tensor Shape
  const silt::vec2 scale,               //!< Cell Scale             [m]           
  const float lambda_max,               //!< Maximum Char. Time     [s]
  const float epsilon                   //!< Minimum Attenuation    [1]
) {

  // Note: Number of concurrent samples given
  //  by the dimensionality of the rng tensor.
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= rng.elem()) return;

  // Extract Correct Dimension Views
  using vecD = silt::fvec<D>;               //!< Vector of Dimension
  auto fluxView = flux.view<vecD>();        //!< Tensorised Flux View
  auto sourceView = source.view<vecD>();    //!< Tensorised Source View
  auto flowView = flow.view<silt::vec2>();  //!< Tensorised Flow View

  // Initialize Particle State
  float att = 1.0f;
  float lambda = 0.0f;
  silt::vec2 pos = silt::vec2 {
    curand_uniform(&rng[n])*float(shape[0]),
    curand_uniform(&rng[n])*float(shape[1])
  };
  int ind = shape.flatten(pos);

  // Upstream Contributions Sample
  const float P = 1.0f / float(shape.elem);
  const vecD S = sourceView[ind] / P;

  // the flow evolution rule?
  //  if we don't have any type of momentum, then pits basically
  //  don't really go away. We rely on the well-structuredness of
  //  the velocity field, which we cannot always do.
  const sample_t<silt::vec2, 2, 1> v_support = sample_t<silt::vec2, 2, 1>::gather(flowView, silt::ivec2(shape[0], shape[1]), pos);
  silt::vec2 v = v_support.val();

  // Integrate along Streamline
  int step = 0;
  while(!shape.oob(pos) && (lambda < lambda_max) && (abs(att) > epsilon) && ++step < 1024) {

    // Accumulate Estimate
    const int nind = shape.flatten(pos);  // New Index?
    if(nind != ind) {
      ind = nind;
      if constexpr(D >= 1){
        atomicAdd(&fluxView[ind].x, S.x * att);
      }
      if constexpr(D >= 2){
        atomicAdd(&fluxView[ind].y, S.y * att);
      }
    }

    const sample_t<silt::vec2, 2, 1> v_support = sample_t<silt::vec2, 2, 1>::gather(flowView, silt::ivec2(shape[0], shape[1]), pos);
    v = 0.75f * v + 0.25f * v_support.val();
    if(glm::length(v) < 1E-8)
      break;

    const float dlambda = glm::length(scale / v);
    att *= __expf(-dlambda * decay[ind]);
    pos += v / glm::length(v);
    lambda += dlambda;

  }

}

template<size_t D>
__global__ void __normalize (
  silt::tensor_t<float> flux,           //!< Flux Integral Estimate []
  const silt::tensor_t<float> flow,     //!< Flow-Field Tensor      []
  const silt::tensor_t<float> source,   //!< Source Term 
  const silt::vec2 scale,               //!< Cell Scale
  const size_t count                    //!< Sample Count
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= flux.elem()/D) return;

  using vecD = silt::fvec<D>;               //!< Vector of Dimension
  auto fluxView = flux.view<vecD>();        //!< Tensorised Flux View
  auto sourceView = source.view<vecD>();    //!< Tensorised Source View
  auto flowView = flow.view<silt::vec2>();  //!< Tensorised Flow View

  const silt::vec2 v = flowView[n];
  fluxView[n] = sourceView[n] + fluxView[n] / float(count) / (scale.x * scale.y);

}

namespace soil {

//! Uniform Distribution Grid-Free Monte-Carlo
//!   Estimator for Linear Conservation Laws
//!
//! Note that this estimator is parameterized so that the 3rd dimension
//! of the source tensor is the dimensionality of the transported quantity.
//!
silt::tensor solve_uniform (
  const silt::tensor_t<float> flow,     //!< Flow-Field Tensor      [m/s x 2]
  const silt::tensor_t<float> source,   //!< Source Term Tensor     [X/s x D]
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor      [1/s x 1]
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source   []
  const silt::vec2 scale,               //!< Cell Scale             [m x 2]
  const size_t count                    //!< Sample Count           []
) {

  // Simulation Parameters
  const float epsilon = 1e-3;   //!< Minimum Attenuation [1]
  const float lambda_max = 128; //!< Maximum Quasi-Static Time [s]

  // The Accumulated Flux has the same Dimension as the Source-Term.
  //  The entirety of the flux is then initialized to zero.
  //  The actual dimension of the domain doesn't have the third component.
  const silt::shape shapeIn = source.shape();
  const size_t D = shapeIn[2];  // Dimensionality of Transported Quantity
  const silt::shape shape = silt::shape(shapeIn[0], shapeIn[1]);
  auto flux = silt::tensor_t<float>(shapeIn, silt::host_t::GPU);
  silt::set(flux, 0.0f);

  // Resolve Data-Layout ofw Source / Flux Tensor
  switch(D) {
    case 1:
      __solve_uniform<1><<<block(rng.elem(), 512), 512>>>(flux, flow, source, decay, rng, shape, scale, lambda_max, epsilon);
      __normalize<1><<<block(flux.elem(), 512), 512>>>(flux, flow, source, scale, count);
      break;
    case 2:
      __solve_uniform<2><<<block(rng.elem(), 512), 512>>>(flux, flow, source, decay, rng, shape, scale, lambda_max, epsilon);
      __normalize<2><<<block(flux.elem(), 512), 512>>>(flux, flow, source, scale, count);
      break;
    default:
      break;
  }

  return silt::tensor(flux);

}

}

/*
//! Mass Deposition Rate [m/y]
__device__ float deposit(const param_t& param, const float dt, const float mass, const float discharge){

  if(discharge < 1.0f)
    return 0.0f;

  const float kd = param.depositionRate;  //!< Fluvial Deposition Rate [m/y]
  return dt * kd * mass / discharge;

}
*/

__global__ void __suspend (
  silt::tensor_t<float> source,
  const silt::tensor_t<float> flow,     //!< Flow-Field Tensor      []
  const silt::vec2 scale,               //!< Cell Scale
  const float ks
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= source.elem()) return;

  auto flowView = flow.view<silt::vec2>();  //!< Tensorised Flow View
  const silt::vec2 speed = flowView[n];

  const float alpha = 0.01f;//param.fluvialExponent;
  const float fD = 0.02f;//param.frictionFactor;      //!< Darcy-Weisbach Friction Factor
  const float rho = 1000.0f;//param.fluvialDensity;     //!< Density of Fluid [kg/m^3]
  //const float ks = 8E-7f;// param.suspensionRate;      //!< Fluvial Suspension Rate [(m^3/y)^-0.4]
  
  const float velocity = glm::length(speed);                    //!< [m/s]
  const float shear = 0.125f * fD * rho * velocity * velocity;  //!< [kg/m/s^2]
  const float power = pow(shear * velocity, alpha);             //!< Stream Power Function
  const float suspend = ks * power;

  source[n] = glm::abs(suspend);

}

namespace soil {

silt::tensor suspend (
  const silt::tensor_t<float> flow,     //!< Flow-Field Tensor
  const silt::vec2 scale,               //!< Cell Scale
  const float ks
) {

  const silt::shape shapeIn = flow.shape();
  const silt::shape shape = silt::shape(shapeIn[0], shapeIn[1]);

  auto source = silt::tensor_t<float>(shape, silt::host_t::GPU);
  __suspend<<<block(source.elem(), 512), 512>>>(source, flow, scale, ks);
  return silt::tensor(source);

}

}

#endif