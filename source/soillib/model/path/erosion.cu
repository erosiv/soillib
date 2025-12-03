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

__global__ void __erode (
  silt::tensor_t<float> height,
  silt::tensor_t<float> discharge,
  silt::tensor_t<float> dischargeTrack,
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

  // Random Sample Position
  silt::vec2 pos = silt::vec2 {
    0.5f + curand_uniform(&rng[n])*float(shape[0] - 1),
    0.5f + curand_uniform(&rng[n])*float(shape[1] - 1)
  };
  int ind = shape.flatten(pos);
  // const float P = 1.0f / float(shape.elem);
  // const vecD S = sourceView[ind] / P;

  // Transport Initialization
  float water = 1.0f;
  float mass = 0.0f;
  silt::vec2 speed = -__grad(height, shape, scale, pos, param.exitSlope);
  if(glm::length(speed) == 0.0f)
    return;

  // Iterate over Number of Steps
  for(int step = 0; step < param.maxage; ++step) {

    // Erosion Step / Mass Integration Step
    const float slope = __slope(height, shape, scale, pos, param.exitSlope);
    const float alpha = param.fluvialExponent;
    const float fD = param.frictionFactor;                        //!< Darcy-Weisbach Friction Factor
    const float rho = mp.density;                                 //!< Density of Fluid [kg/m^3]
    const float ks = param.suspensionRate;                        //!< Fluvial Suspension Rate [(m^3/y)^-0.4
    const float velocity = glm::length(speed);                    //!< [m/s]
    const float shear = 0.125f * fD * rho * velocity * velocity;  //!< [kg/m/s^2]
    const float power = pow(shear * velocity, alpha);             //!< Stream Power Function

    const float suspend = glm::max(0.0f, ks * power * slope);
    const float deposit = glm::min(mass, param.depositionRate * mass / water);
    const float transfer = suspend - deposit;

    atomicAdd(&height[ind], -transfer / scale.z);
    mass += transfer;
    water = (1.0f - param.evapRate) * water;

    // Position Update
    pos += speed / glm::length(speed);
    if(__oob(shape, pos))
      break;

    // Tracking Step
    ind = shape.flatten(pos);
    atomicAdd(&dischargeTrack[ind], water);
    atomicAdd(&momentumTrackView[ind].x, water * speed.x);
    atomicAdd(&momentumTrackView[ind].y, water * speed.y);

    // Velocity Update
    const silt::vec2 mspeed = momentumView[ind] / discharge[ind];

    const float nu = mp.viscosity;
    const float tau = mp.bedShear;

    speed = (1.0f - tau) * speed;
    speed = ((1.0f - nu) * speed + nu * mspeed);
    speed = (speed - __grad(height, shape, scale, pos, param.exitSlope));
    if(glm::length(speed) < 1E-6f)
      break;

  }

}

void soil::erode (
  silt::tensor_t<float> height,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentumTrack,
  silt::tensor_t<float> discharge,
  silt::tensor_t<float> dischargeTrack,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
) {

  // velocity field?
  const float A = scale.x * scale.y;

  silt::set(dischargeTrack, A);
  silt::set(momentumTrack, 0.0f);

  __erode<<<block(rng.elem(), 512), 512>>> (
    height,
    discharge,
    dischargeTrack,
    momentum.view<silt::vec2>(),
    momentumTrack.view<silt::vec2>(),
    rng,
    height.shape(),
    scale,
    param,
    mp
  );

  silt::mix(discharge, dischargeTrack, param.lrate);
  silt::mix(momentum, momentumTrack, param.lrate);

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
    curand_uniform(&rng[n])*float(shape[0]),
    curand_uniform(&rng[n])*float(shape[1])
  };
  int ind = shape.flatten(pos);
  const float P = 1.0f / float(shape.elem);
  // const vecD S = sourceView[ind] / P;

  // Transport Initialization
  silt::vec2 speed = -__grad(height, shape, scale, pos, param.exitSlope);
  float mass = 0.0f;

  // Iterate over Number of Steps
  for(int step = 0; step < param.maxage; ++step) {

    // Debris-Flow Erosion Formula
    const float slope = __slope(height, shape, scale, pos, param.exitSlope);
    
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

    speed = (1.0f - tau) * speed;
    speed = ((1.0f - nu) * speed + nu * mspeed);
    speed = (speed - __grad(height, shape, scale, pos, param.exitSlope));
    if(glm::length(speed) < 1E-6f)
      break;
  
  }

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

  // velocity field?
  const float A = scale.x * scale.y;

  silt::set(massTrack, 0.0f);
  silt::set(momentumTrack, 0.0f);

  const silt::shape shapeIn = height.shape();

  __erode_debris<<<block(rng.elem(), 512), 512>>> (
    height,
    mass,
    massTrack,
    momentum.view<silt::vec2>(),
    momentumTrack.view<silt::vec2>(),
    rng,
    shapeIn,
    scale,
    param,
    mp
  );

  silt::mix(mass, massTrack, param.lrate);
  silt::mix(momentum, momentumTrack, param.lrate);

}

#endif