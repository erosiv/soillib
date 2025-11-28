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
    v = v_support.val();
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

__device__ silt::vec2 __grad(
  const silt::tensor_t<float>& height,
  const silt::shape shape,
  const silt::vec3 scale,
  const silt::vec2 pos
){

  const int i00 = shape.flatten(pos + silt::vec2( 0, 0));
  const int in0 = shape.flatten(pos + silt::vec2(-1, 0));
  const int ip0 = shape.flatten(pos + silt::vec2( 1, 0));
  const int i0n = shape.flatten(pos + silt::vec2( 0,-1));
  const int i0p = shape.flatten(pos + silt::vec2( 0, 1));

  const float h = height[i00] * scale.z;
  const float hn0 = shape.oob(pos + silt::vec2(-1, 0)) ? h : height[in0] * scale.z;
  const float hp0 = shape.oob(pos + silt::vec2( 1, 0)) ? h : height[ip0] * scale.z;
  const float h0n = shape.oob(pos + silt::vec2( 0,-1)) ? h : height[i0n] * scale.z;
  const float h0p = shape.oob(pos + silt::vec2( 0, 1)) ? h : height[i0p] * scale.z;

  float gx = 0.0f;
  if(__isnanf(hn0)) gx = (hp0 - h)/scale.x;
  if(__isnanf(hp0)) gx = (h - hn0)/scale.x;
  if(!__isnanf(hp0) && !__isnanf(hn0)){
    if(hn0 < hp0) gx = (h - hn0)/scale.x;
    if(hp0 < hn0) gx = (hp0 - h)/scale.x;
    // if they are the same, slope is zero
  }

  float gy = 0.0f;
  if(__isnanf(h0n)) gy = (h0p - h)/scale.y;
  if(__isnanf(h0p)) gy = (h - h0n)/scale.y;
  if(!__isnanf(h0p) && !__isnanf(h0n)){
    if(h0n < h0p) gy = (h - h0n)/scale.y;
    if(h0p < h0n) gy = (h0p - h)/scale.y;
    // if they are the same, slope is zero
  }

  return silt::vec2(gx, gy);

}

__global__ void __erode (
  silt::tensor_t<float> height,
  silt::tensor_t<float> discharge,
  silt::tensor_t<float> dischargeTrack,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentumTrack,
  silt::tensor_t<silt::rng> rng,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= rng.elem()) return;

  // Random Sample Position
  silt::vec2 pos = silt::vec2 {
    curand_uniform(&rng[n])*float(shape[0]),
    curand_uniform(&rng[n])*float(shape[1])
  };
  int ind = shape.flatten(pos);
  const float P = 1.0f / float(shape.elem);

  auto momentumView = momentum.view<silt::vec2>();
  auto momentumTrackView = momentumTrack.view<silt::vec2>();

  // const vecD S = sourceView[ind] / P;
  // Transport Initialization
  
  float water = 1.0f;
  float sed = 0.0f;
  silt::vec2 speed = -__grad(height, shape, scale, pos);
  float height_cur = height[ind] * scale.z;

  int step = 0;
  while(!shape.oob(pos) && ++step < param.maxage) {

    // Position Update
    if(glm::length(speed) < 1E-6f)
      break;

    pos += speed / glm::length(speed);
    if(shape.oob(pos))
      break;
    const int nind = shape.flatten(pos);

    // Tracking Step
    atomicAdd(&dischargeTrack[nind], water);
    atomicAdd(&momentumTrackView[nind].x, water * speed.x);
    atomicAdd(&momentumTrackView[nind].y, water * speed.y);

    // Velocity Update
    const silt::vec2 mspeed = momentumView[ind] / discharge[ind];

    const float nu = param.viscosity;
    const float tau = param.bedShear;

    speed = (1.0f - tau) * speed;
    speed = ((1.0f - nu) * speed + nu * mspeed);
    speed = (speed - __grad(height, shape, scale, pos));

    // Erosion Update
    const float height_next = height[nind] * scale.z;
    const float alpha = param.fluvialExponent;
    const float fD = param.frictionFactor;      //!< Darcy-Weisbach Friction Factor
    const float rho = param.fluvialDensity;     //!< Density of Fluid [kg/m^3]
    const float ks = param.suspensionRate;      //!< Fluvial Suspension Rate [(m^3/y)^-0.4]

//    const float velocity = glm::length(speed);                    //!< [m/s]
//    const float shear = 0.125f * fD * rho * velocity * velocity;  //!< [kg/m/s^2]
//    const float power = pow(shear * velocity, alpha);             //!< Stream Power Function
//    const float suspend = glm::max(0.0f, ks * power * (height_cur - height_next));
    const float suspend = glm::max(0.0f, param.suspensionRate * (height_cur - height_next));

    const float deposit = glm::min(sed, param.depositionRate * sed / water);
    const float transfer = suspend - deposit;
    atomicAdd(&height[ind], -transfer / scale.z);
    sed += transfer;

    water = (1.0f - param.evapRate) * water;

    height_cur = height_next;
    ind = nind;

  }

}

__global__ void __erode_debris (
  silt::tensor_t<float> height,
  silt::tensor_t<float> massBuf,
  silt::tensor_t<float> massTrack,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentumTrack,
  silt::tensor_t<silt::rng> rng,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= rng.elem()) return;

  // Random Sample Position
  silt::vec2 pos = silt::vec2 {
    curand_uniform(&rng[n])*float(shape[0]),
    curand_uniform(&rng[n])*float(shape[1])
  };
  int ind = shape.flatten(pos);
  const float P = 1.0f / float(shape.elem);
  // const vecD S = sourceView[ind] / P;

//  auto momentumView = momentum.view<silt::vec2>();
//  auto momentumTrackView = momentumTrack.view<silt::vec2>();

  // Transport Initialization
  float mass = 0.0001f;
  silt::vec2 speed = -__grad(height, shape, scale, pos);
  float height_cur = height[ind] * scale.z;

  int step = 0;
  while(!shape.oob(pos) && ++step < param.maxage) {

    // Position Update
    if(glm::length(speed) < 1E-6f)
      break;

    pos += speed / glm::length(speed);
    if(shape.oob(pos))
      break;
    const int nind = shape.flatten(pos);

    // Tracking Step
    atomicAdd(&massTrack[nind], mass);
//    atomicAdd(&momentumTrackView[nind].x, mass * speed.x);
//    atomicAdd(&momentumTrackView[nind].y, mass * speed.y);

    // Velocity Update
//    const silt::vec2 mspeed = momentumView[ind] / mass[ind];

//    const float nu = param.viscosity;
    const float tau = param.bedShear;

    speed = (1.0f - tau) * speed;
//    speed = ((1.0f - nu) * speed + nu * mspeed);
    speed = (speed - __grad(height, shape, scale, pos));

    // Erosion Update
    const float height_next = height[nind] * scale.z;

    // Debris-Flow Erosion Formula

    const float slope = (height_cur - height_next);
    const float suspend = param.debrisSuspensionRate * glm::max(0.0f, slope - param.critSlope);
    const float deposit = glm::min(mass, param.debrisDepositionRate * mass);
    const float transfer = suspend - deposit;
    atomicAdd(&height[ind], -transfer / scale.z);
    mass += transfer;

//    const float shearViscous = param.debrisViscosity * glm::length(speed) / (0.0001f + mass);
//    const float shearYield = param.debrisYieldStress + mass * param.critSlope;
//    const float shearMass = mass * (height_cur - height_next);
//    const float suspend = glm::max(0.0f, param.debrisSuspensionRate * (shearMass - shearViscous - shearYield));
//    const float deposit =  glm::min(mass, glm::max(0.0f, param.debrisDepositionRate * (shearViscous + shearYield - shearMass)));
//
////    const float deposit = glm::min(sed, param.depositionRate * sed / water);

    // Update Tracking Variables
    height_cur = height_next;
    ind = nind;

  }

}

namespace soil {

void erode (
  silt::tensor_t<float> height,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentumTrack,
  silt::tensor_t<float> discharge,
  silt::tensor_t<float> dischargeTrack,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param
) {

  // velocity field?
  const float A = scale.x * scale.y;

  silt::set(dischargeTrack, A);
  silt::set(momentumTrack, 0.0f);

  __erode<<<block(rng.elem(), 512), 512>>>(
    height,
    discharge,
    dischargeTrack,
    momentum,
    momentumTrack,
    rng, height.shape(), scale, param);

  silt::mix(discharge, dischargeTrack, param.lrate);
  silt::mix(momentum, momentumTrack, param.lrate);

}

void erode_debris (
  silt::tensor_t<float> height,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentumTrack,
  silt::tensor_t<float> mass,
  silt::tensor_t<float> massTrack,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param
) {

  // velocity field?
  const float A = scale.x * scale.y;

  silt::set(massTrack, A);
  silt::set(momentumTrack, 0.0f);

  __erode_debris<<<block(rng.elem(), 512), 512>>>(
    height,
    mass,
    massTrack,
    momentum,
    momentumTrack,
    rng, height.shape(), scale, param);

  silt::mix(mass, massTrack, param.lrate);
  silt::mix(momentum, momentumTrack, param.lrate);

}

}

#endif