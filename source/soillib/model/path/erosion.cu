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

__global__ void __transport_fluvial (
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
  const float rho_w = mp.density;         //!< Density of Water     [kg/m^3]
  const float rho_s = mp.density * 2.0f;  //!< Density of Sediment  [kg/m^3]
  const float R = 1.0f;                   //!< Rainfall Rate        [m/y]

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

__global__ void __transport_debris (
  const silt::tensor_t<float> height,
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

  const float A = scale.x * scale.y;      //!< Cell Area            [m^2]

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
  silt::vec2 speed = - (mp.gravity * grad);

  const float slopeExcess = (glm::length(grad) - param.critSlope);
  const float suspend = fmaxf(0.0f, param.debrisViscousStress * slopeExcess - param.debrisYieldStress);
  float vol_d = A * Q * suspend;

  // Iterate over Number of Steps
  for(int step = 0; step < param.maxage; ++step) {

    // Debris-Flow Erosion Formula
    const float slope = glm::length(grad);
    const float shearDebris = vol_d * (slope - param.critSlope) - param.debrisYieldStress;
    const float suspend = param.debrisSuspensionRate * shearDebris;
    const float deposit = fmaxf(param.debrisDepositionRate * shearDebris, -vol_d);
    if(shearDebris < 0.0f)  {
      vol_d = vol_d + deposit;
    } else {
      vol_d = vol_d + suspend;
    }

//    if(debrisGrowth > 0.0f) debrisGrowth *= param.debrisSuspensionRate;
//    else debrisGrowth *= param.debrisDepositionRate;
//
    // finite growth rate...
    // of course, in reality it should be so that we can't go
    //  beyond the excess of the slope...
    //  so debrisgrowth * vol_d has to be less than 1... 
    // debrisGrowth = fminf(fmaxf(debrisGrowth, -1.0f), 1.0f);
    // so how do we stop it from running away?
    //  I suppose we could think of the steady-state amount...

    
    // Equilibrium Mass Limiting...
//    if(vol_d > shearLandslide) vol_d = shearLandslide;

    // Position Update
    pos += speed / glm::length(speed);
    if(__oob(shape, pos))
      break;
    
    // Tracking Step
    const int nind = shape.flatten(pos);
    if(nind != ind){
      atomicAdd(&massTrack[nind], vol_d);
      atomicAdd(&momentumTrack[nind].x, vol_d * speed.x);
      atomicAdd(&momentumTrack[nind].y, vol_d * speed.y);
      ind = nind;
    }

    // Velocity Update
    const silt::vec2 mspeed = __divzero(momentum[ind], massBuf[ind]);
    const float nu = mp.viscosity;
    const float tau = mp.bedShear;

    grad = __grad(height, shape, scale, pos, param.exitSlope);
    speed = (1.0f - tau) * speed;
    speed = ((1.0f - nu) * speed + nu * mspeed);
    speed = (speed - mp.gravity * grad);
    if(glm::length(speed) < 1E-6f)
      break;
  
  }

}

//! Mass-Transfer Execution Kernel
//! This kernel edits the height-field and material distributions.
//!
//! Dimensionalization Notes:
//!   - The height-field is stored as a dimensionless quantity, and
//!     can be converted to meters using the scale.z parameter.
//!   - The transported fields have to add the source terms explicitly,
//!     because not every position generates a sample. This is accounted
//!     for in the transport kernels.
//!
__global__ void __transfer (
  silt::tensor_t<float> height,
  const silt::tensor_t<float> upliftBase,
  const silt::tensor_t<float> discharge,
  const silt::tensor_t<float> mass,
  const silt::const_view_t<silt::vec2> momentumFluvial,
  const silt::tensor_t<float> debris,
  const silt::const_view_t<silt::vec2> momentumDebris,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= height.elem())
    return;

  // Compute Local Properties
  const silt::vec2 pos = shape.unflatten(n);
  const silt::vec2 grad = __grad(height, shape, scale, pos, param.exitSlope); // []
  const float L = glm::length(glm::vec2(scale.x, scale.y));                   // [m]
  const silt::vec2 speed = momentumFluvial[n] / discharge[n] - (mp.gravity * grad);
  const float conc = mass[n] / discharge[n];
  const float slope = glm::length(grad);                                      // []

  // General Dimensionalized Parameters
  const float dt = param.timeStep;            // Simulation Timestep            [y]
  const float ku = param.uplift;              // Terrain Uplift Rate            [m/y]
  const float kfs = param.suspensionRate;     // Fluvial Suspension Rate
  const float kfd = param.depositionRate;     // Fluvial Deposition Rate
  const float fD = param.frictionFactor;      // Darcy-Weisbach Friction Factor []
  const float alpha = param.fluvialExponent;  // Power Law Exponent             []
  const float density = mp.density;           // Fluid Density                  [kg/m^3]

  // Fluvial Erosion Computation
  const float vel = glm::length(speed);                     // Fluid Velocity           [m/s]
  const float shear = 0.125f * fD * density * vel * vel;    // Wall Shear Stress        [kg/m/s^2 = Pa]
  const float power = glm::abs(__powf(shear * vel, alpha)); // Stream Power Function    [(kg/s^3)^a]
  const float suspend = kfs * power;                        // Fluvial Suspension Rate  [m/y]
  const float deposit = kfd * conc;                         // Fluvial Deposition Rate  [m/y]
  const float uplift = ku * upliftBase[n];                  // Terrain Uplift Rate      [m/y]

  // Debris Erosion Computation  
  const float shearLandslide = param.debrisViscousStress * fmaxf(0.0f, slope - param.critSlope - param.debrisYieldStress);
  const float shearYield = debris[n] * (slope - param.critSlope) - param.debrisYieldStress;
  const float suspendDebris = fminf(0.01f * slope * L, shearLandslide + param.debrisSuspensionRate * fmaxf(0.0f, shearYield));
  const float depositDebris = fminf(debris[n], fmaxf(0.0f, -param.debrisDepositionRate * shearYield));

  // Height-Field Update []
  height[n] += dt * (uplift + deposit - suspend * slope) / scale.z;
  height[n] += dt * (depositDebris - suspendDebris) / scale.z;

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

void soil::transport_fluvial (
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

  __transport_fluvial<<<block(rng.elem(), 512), 512>>> (
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

void soil::transport_debris (
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

  __transport_debris<<<block(rng.elem(), 512), 512>>> (
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
  const silt::tensor_t<float> momentumFluvial,
  const silt::tensor_t<float> debris,
  const silt::tensor_t<float> momentumDebris,
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
    momentumFluvial.view<silt::vec2>(),
    debris,
    momentumDebris.view<silt::vec2>(),
    shape, scale, param, mp
  );

}

#endif