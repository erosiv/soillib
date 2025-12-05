#ifndef SOILLIB_MODEL_EROSION_CU
#define SOILLIB_MODEL_EROSION_CU
#define HAS_CUDA

#include <soillib/soillib.hpp>

#include <silt/core/error.hpp>
#include <silt/core/tensor.hpp>
#include <silt/op/gather.hpp>
#include <silt/op/common.hpp>

#include <soillib/model/path/erosion.hpp>
#include <soillib/model/path/erosion_map.cu>

#include <math_constants.h>

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

__device__ float __source_sediment(
  const silt::vec2 grad,
  const silt::vec2 speed,
  const soil::param_t param,
  const soil::momentum_param_t mp
) {

  // Erosion Step / Mass Integration Step
  const float slope = glm::length(grad);
  const float fD = param.frictionFactor;                    //!< Darcy-Weisbach Friction Factor
  const float alpha = param.fluvialExponent;
  const float density = mp.density;                         //!< Total Density
  const float vel = glm::length(speed);                     //!< [m/s]
  const float shear = 0.125f * fD * density * vel * vel;    //!< [kg/m/s^2]
  const float power = glm::abs(__powf(shear * vel, alpha)); //!< Stream Power Function
  const float suspend = glm::max(0.0f,  param.suspensionRate * power * slope);
  return suspend;

}

__global__ void __erode (
  silt::tensor_t<float> height,
  silt::tensor_t<float> discharge,
  silt::tensor_t<float> dischargeTrack,
  silt::tensor_t<float> mass,
  silt::tensor_t<float> massTrack,
  silt::view_t<silt::vec2> momentumView,
  silt::view_t<silt::vec2> momentumTrackView,
  silt::tensor_t<silt::rng> rng,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= rng.elem()) return;

  const float A = scale.x * scale.y;      //!< Cell Area            [m^2]
  const float rho_w = mp.density;         //!< Density of Water     [g / m^3]
  const float rho_s = mp.density * 2.0f;  //!< Density of Sediment  [g / m^3]
  const float R = 1.0f;                   //!< Rainfall Rate        [m / y]

  // Sampling Procedure
  silt::vec2 pos = silt::vec2 {
    0.5f + curand_uniform(&rng[n])*float(shape[0] - 1),
    0.5f + curand_uniform(&rng[n])*float(shape[1] - 1)
  };
  int ind = __flatten(shape, pos);
  const float N = rng.elem();
  const float Q = float(shape.elem) / N;  // Scaling Factor

  // Transport Initialization
  silt::vec2 grad = __grad(height, shape, scale, pos, param.exitSlope);
  silt::vec2 speed = - (mp.gravity * grad);
  float vol_w = A * Q * R;                                          //!< Sampled Water Source Term
  float vol_s = A * Q * __source_sediment(grad, speed, param, mp);  //!< Sampled Sediment Source Term

  if(glm::length(speed) == 0.0f)
    return;

  // Iterate over Number of Steps
  for(int step = 0; step < param.maxage; ++step) {

    //! Attenuate the Sampled Transport Quantities
//    vol_s = vol_s - glm::min(vol_s, param.depositionRate * vol_s / vol_w);
    vol_w = (1.0f - param.evapRate) * vol_w;

    // Position Update
    pos += speed / glm::length(speed);
    if(__oob(shape, pos))
      break;

    // Tracking Step
    const int nind = __flatten(shape, pos);
    if(nind != ind){
      atomicAdd(&dischargeTrack[nind], vol_w);
      atomicAdd(&massTrack[nind], vol_s);
      atomicAdd(&momentumTrackView[nind].x, vol_w * speed.x);
      atomicAdd(&momentumTrackView[nind].y, vol_w * speed.y);
      ind = nind;
    }

    // Velocity Update
    const silt::vec2 mspeed = momentumView[ind] / discharge[ind];

    const float nu = mp.viscosity;
    const float tau = mp.bedShear;
    
    grad = __grad(height, shape, scale, pos, param.exitSlope);
    speed = ((1.0f - nu) * speed + nu * mspeed);
    // basically, the momentum always wins?
//    speed = ((1.0f - nu) * mass * speed + nu * momentumView[ind]) / ((1.0f - nu) * mass + nu * discharge[ind]);
    speed = (1.0f - tau) * speed; // Self-Drag Application
    speed = (speed - mp.gravity * grad);
    if(glm::length(speed) < 1E-6f)
      break;

  }

}

__global__ void __erode_debris (
  silt::tensor_t<float> height,
  silt::tensor_t<float> massBuf,
  silt::tensor_t<float> massTrack,
  silt::view_t<silt::vec2> momentum,
  silt::view_t<silt::vec2> momentumTrack,
  silt::tensor_t<silt::rng> rng,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= rng.elem()) return;

  // Random Sample Position
  silt::vec2 pos = silt::vec2 {
    0.5f + curand_uniform(&rng[n])*float(shape[0] - 1),
    0.5f + curand_uniform(&rng[n])*float(shape[1] - 1)
  };
  int ind = shape.flatten(pos);
  const float N = rng.elem();
  const float Q = float(shape.elem) / N;  // Scaling Factor

  // Transport Initialization
  silt::vec2 grad = __grad(height, shape, scale, pos, param.exitSlope);
  silt::vec2 speed = -grad;
  float mass = 0.0f;

  // Iterate over Number of Steps
  for(int step = 0; step < param.maxage; ++step) {

    // Debris-Flow Erosion Formula
    const float slope = glm::length(grad);
    const float shearLandslide = glm::max(0.0f, slope - param.critSlope);
    // note: this implies the existence of the landslide mass...
    const float shearViscous = param.debrisViscousStress * glm::length(speed) / (1.0f + mass);
    const float shearYield = mass * (slope - param.critSlope) - param.debrisYieldStress;
    const float suspend = glm::max(0.0f, param.debrisSuspensionRate * (shearLandslide + shearYield - shearViscous));
    const float deposit = glm::min(mass, glm::max(0.0f, param.debrisDepositionRate * (shearViscous - shearYield - shearLandslide)));

    const float transfer = suspend - deposit;
    atomicAdd(&height[ind], -transfer / scale.z);
    mass += transfer;

    // Position Update
    pos += speed / glm::length(speed);
    if(__oob(shape, pos))
      break;
    
    // Tracking Step
    ind = shape.flatten(pos);
    atomicAdd(&massTrack[ind], mass);
    atomicAdd(&momentumTrack[ind].x, mass * speed.x);
    atomicAdd(&momentumTrack[ind].y, mass * speed.y);

    // Velocity Update
    const silt::vec2 mspeed = momentum[ind] / (1.0f + massBuf[ind]);
    const float nu = mp.viscosity;
    const float tau = mp.bedShear;

    grad = __grad(height, shape, scale, pos, param.exitSlope);
    speed = (1.0f - tau) * speed;
    speed = ((1.0f - nu) * speed + nu * mspeed);
    speed = (speed - grad);
    if(glm::length(speed) < 1E-6f)
      break;
  
  }

}

__global__ void __transfer (
  silt::tensor_t<float> height,
  const silt::tensor_t<float> upliftBase,
  const silt::tensor_t<float> discharge,
  const silt::tensor_t<float> mass,
  const silt::const_view_t<silt::vec2> momentumView,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
) {

  // ... execute the mass transfer on height-map ...
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= height.elem()) return;

  const silt::vec2 pos = shape.unflatten(n);
  const silt::vec2 grad = __grad(height, shape, scale, pos, param.exitSlope);
  // basically take the source component + the cumulative component...
  const silt::vec2 speed = momentumView[n] / discharge[n] - (mp.gravity * grad);
  const float conc = mass[n] / discharge[n];
  const float slope = glm::length(grad);

  // Erosion Step / Mass Integration Step
  const float fD = param.frictionFactor;                    //!< Darcy-Weisbach Friction Factor
  const float alpha = param.fluvialExponent;
  const float density = mp.density;                         //!< Total Density
  const float vel = glm::length(speed);                     //!< [m/s]
  const float shear = 0.125f * fD * density * vel * vel;    //!< [kg/m/s^2]
  const float power = glm::abs(__powf(shear * vel, alpha)); //!< Stream Power Function
  const float suspend = param.suspensionRate * power;
  const float deposit = param.depositionRate * conc;
  const float uplift = param.uplift * upliftBase[n];

  height[n] += param.timeStep * (uplift + deposit - suspend * slope);

  // rate limiting based on the slope ...
//  hdiff = glm::max(hdiff, - 0.5f * slope * glm::length(silt::vec2(scale.x, scale.y)));
  
  // ... basically we want to do the height-field update based on steady-state approach ...
  //  ... that makes the height-field update unconditionally stable ...
  //  ...

  // basically approach the slope and update the height-field accordingly...
  // ...

//  const float slopeSteady = fminf(fmaxf((uplift + deposit) / (1E-6f + suspend), 0.0f), param.critSlope);
//  const float slopeDiff = (slopeSteady - slope) * (1.0f - expf(-(1E-6f + suspend) * param.timeStep));
//  height[n] += slopeDiff * glm::length(silt::vec2(scale.x, scale.y));

  // if timestep approaches infinity, then the slope difference becomes
  //  equal to the difference to the steady-state. Otherwise,
  //  it is exactly zero and there is no update.

}

//
// Kernel Launch Implementations
//

void soil::erode (
  silt::tensor_t<float> height,
  silt::tensor_t<float> discharge,
  silt::tensor_t<float> dischargeTrack,
  silt::tensor_t<float> mass,
  silt::tensor_t<float> massTrack,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentumTrack,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
) {

  const float A = scale.x * scale.y;
  const silt::shape shape = height.shape();

  silt::set(dischargeTrack, A);
  silt::set(massTrack, 0.0f);
  silt::set(momentumTrack, 0.0f);

  __erode<<<block(rng.elem(), 512), 512>>> (
    height,
    discharge,
    dischargeTrack,
    mass,
    massTrack,
    momentum.view<silt::vec2>(),
    momentumTrack.view<silt::vec2>(),
    rng, shape, scale, param, mp
  );

  silt::mix(discharge, dischargeTrack, param.lrate);
  silt::mix(mass, massTrack, param.lrate);
  silt::mix(momentum, momentumTrack, param.lrate);

}

void soil::erode_debris (
  silt::tensor_t<float> height,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentumTrack,
  silt::tensor_t<float> mass,
  silt::tensor_t<float> massTrack,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
) {

  const float A = scale.x * scale.y;
  const silt::shape shape = height.shape();

  silt::set(massTrack, 0.0f);
  silt::set(momentumTrack, 0.0f);

  __erode_debris<<<block(rng.elem(), 512), 512>>> (
    height,
    mass,
    massTrack,
    momentum.view<silt::vec2>(),
    momentumTrack.view<silt::vec2>(),
    rng, shape, scale, param, mp
  );

  silt::mix(mass, massTrack, param.lrate);
  silt::mix(momentum, momentumTrack, param.lrate);

}

void soil::mass_transfer (
  silt::tensor_t<float> height,
  const silt::tensor_t<float> uplift,
  const silt::tensor_t<float> discharge,
  const silt::tensor_t<float> mass,
  const silt::tensor_t<float> momentum,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
) {

  const silt::shape shape = height.shape();
  
  __transfer<<<block(height.elem(), 512), 512>>> (
    height,
    uplift,
    discharge,
    mass,
    momentum.view<silt::vec2>(),
    shape, scale, param, mp
  );

}

#endif