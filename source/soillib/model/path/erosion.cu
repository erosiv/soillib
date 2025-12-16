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

  // Physical Parameter Set
  const auto A = scale.x * scale.y;             //!< Cell Area                      [m^2]
  const auto L = silt::vec2(scale.x, scale.y);  //!< Cell Width                     [m]
  const auto rho_w = mp.density;                //!< Density of Water               [kg/m^3]
  const auto rho_s = mp.density * 2.0f;         //!< Density of Sediment            [kg/m^3]
  const auto tau = mp.bedShear;                 //!<
  const auto nu = mp.viscosity;                 //!< Kinematic Viscosity            [m^2/s]
  const auto g = mp.gravity;                    //!< Gravitational Acceleration     [m/s^2]
  const auto ks = param.suspensionRate;         //!< Fluvial Suspension Rate
  const auto fD = param.frictionFactor;         //!< Darcy-Weisbach Friction Factor []
  const auto alpha = param.fluvialExponent;     //!< Suspension Power               []
  const auto R = 1.0f;                          //!< Rainfall Rate                  [m/y]
  const auto eps = 1E-12f;                      //!< Attenuation Threshold          []

  // Sampling Procedure
  const float N = rng.elem();                   //!< Total Sample Count [#]
  const float P = 1.0f / float(shape.elem);     //!< Sample Probability 
  const float Q = 1.0f / (P * N * A);           //!< Normalization Factor (Uniform Grid)
  silt::vec2 pos {                              //!< Sampled Position
    0.5f + curand_uniform(&rng[n])*float(shape[0] - 1),
    0.5f + curand_uniform(&rng[n])*float(shape[1] - 1)
  };
  int ind = __flatten(shape, pos);              //!< Sampled Index

  // Trajectory Initialization
  silt::vec2 grad = __grad(height, shape, scale, pos, param.exitSlope);
  silt::vec2 speed = - (g * grad) + nu * momentumView[ind];
  speed = speed / sqrtf(glm::length(L * speed));
  if(glm::length(speed) < eps)
    return;
    
  // Transport Source / Attenuation Terms
  //  Note that the source terms are always scaled expressions or rate.
  //  

  const float vel = glm::length(speed);                 //!< [m/s]
  const float shear = 0.125f * fD * rho_w * vel * vel;  //!< [kg/m/s^2]

  const auto source_w = Q * R;
  const auto source_m = Q * ks * __powf(shear * vel, alpha);
  const auto source_v = Q * (- (g * grad) + nu * momentumView[ind]);

  float att_w = 1.0f;
  float att_m = 1.0f;
  float att_v = 1.0f;

  // Iterate over Number of Steps
  for(int step = 0; step < param.maxage; ++step) {

    //! Update Transport Attenuation
    float ds = glm::length(silt::vec2(scale.x, scale.y) / speed);
    att_m = att_m * __expf(-ds * param.depositionRate * (source_m) / (att_w * source_w));
    att_w = att_w * __expf(-ds * param.evapRate);
    att_v = att_v * __expf(-ds * (tau + nu));

    //! Velocity Update
    grad = __grad(height, shape, scale, pos, param.exitSlope);
    speed = speed - ds * (mp.gravity * grad) + ds * nu * momentumView[ind]; //!< Particle Velocity Source
    speed = __expf(-ds * (tau + nu)) * speed;                               //!< Particle Velocity Decay
    if(glm::length(speed) < eps)
      break;
    
    // Position Update
    pos += speed / glm::length(speed);
    if(__oob(shape, pos))
      break;

    // Tracking Step
    const int nind = __flatten(shape, pos);
    if(nind != ind) {
      atomicAdd(&dischargeTrack[nind],      att_w * source_w);
      atomicAdd(&massTrack[nind],           att_m * source_m);
      atomicAdd(&momentumTrackView[nind].x, att_v * source_v.x);
      atomicAdd(&momentumTrackView[nind].y, att_v * source_v.y);
      ind = nind;
    }

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
    const float shearDebris = mp.gravity * (vol_d * (slope - param.critSlope) - param.debrisYieldStress);
    const float suspend = param.debrisSuspensionRate * shearDebris;
    const float deposit = fmaxf(param.debrisDepositionRate * shearDebris, -vol_d);
    if(shearDebris < 0.0f)  {
      vol_d = vol_d + deposit;
    } else {
      vol_d = vol_d + suspend;
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
//! Basically, there will be some limiting velocity based on the parameters,
//!   which will depend on the slope. This in turn will lead to some kind of
//!   steady-state slope. We could model this directly. Some dimensional
//!   analysis could help in general with making the model more intuitive.
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
  const silt::vec2 speed = momentumFluvial[n] - (mp.gravity * grad);
  const float conc = mass[n] / discharge[n];
  const float slope = glm::length(grad);                                      // []

  // General Dimensionalized Parameters
  const float dt = param.timeStep;              // Simulation Timestep            [y]
  const float ku = param.uplift;                // Terrain Uplift Rate            [m/y]
  const float kfs = param.suspensionRate;       // Fluvial Suspension Rate
  const float kfd = param.depositionRate;       // Fluvial Deposition Rate
  const float fD = param.frictionFactor;        // Darcy-Weisbach Friction Factor []
  const float alpha = param.fluvialExponent;    // Power Law Exponent             []
  const float density = mp.density;             // Fluid Density                  [kg/m^3]
  const float g = mp.gravity;                   // Gravitational Acceleration     [m/s^2]
  const float tau_y = param.debrisYieldStress;  // Normalized Yield Stress
  const float kL = param.debrisViscousStress;   // Landslide Erosion Rate
  const float kds = param.debrisSuspensionRate; // Debris Suspension Rate
  const float kdd = param.debrisDepositionRate; // Debris Deposition Rate

  // Fluvial Erosion Computation
  const float vel = glm::length(speed);                       // Fluid Velocity           [m/s]
  const float shear = 0.125f * fD * density * vel * vel;      // Wall Shear Stress        [kg/m/s^2 = Pa]
  const float power = glm::abs(__powf(shear * vel, alpha));   // Stream Power Function    [(kg/s^3)^a]
  const float suspend = kfs * power;                          // Fluvial Suspension Rate  [m/y]
  // const float suspend = param.suspensionRate * __powf(discharge[n], 0.4f) * slope;

  const float deposit = kfd * conc;                           // Fluvial Deposition Rate  [m/y]
  const float uplift = ku * upliftBase[n];                    // Terrain Uplift Rate      [m/y]

  // Debris Erosion Computation
  const float debrisHeight = debris[n];                                             // Debris Flow Height [m]
  const float excessSlope = (slope - param.critSlope);                              // Excess Slope []
  const float shearLandslide = fmaxf(0.0f, kL * excessSlope - tau_y);               //
  const float shearYield = g * (debrisHeight * excessSlope - tau_y);                //
  const float suspendDebris = shearLandslide + kds * fmaxf(0.0f, shearYield);       // Debris Suspension Rate [m/y]
  const float depositDebris = fminf(debrisHeight, fmaxf(0.0f, -kdd * shearYield));  // Debris Deposition Rate [m/y]

  // Height-Field Update (Stabilized)
  //  The erosion system is not permitted to generate a pit, because pits become self-reinforcing and numerically
  //  unstable by this model. The model assumes no pits. Therefore, we can use the local slope to limit the erosion
  //  rate. Physically, this can be interpreted as an exponential approach towards the steady-state.

  float transfer = dt * (uplift + deposit - suspend + depositDebris - suspendDebris);
  transfer = fmaxf(transfer, -0.25f * L * slope);
  transfer = fminf(transfer,  0.25f * L * param.critSlope);
  height[n] += transfer / scale.z;

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

  // basically, this set should be initialized correctly...
  // it is currently NOT initialized correctly.  
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