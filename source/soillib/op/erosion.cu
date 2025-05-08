#ifndef SOILLIB_NODE_EROSION_CU
#define SOILLIB_NODE_EROSION_CU
#define HAS_CUDA

#include <soillib/util/error.hpp>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <iostream>

#include <soillib/op/gather.hpp>
#include <soillib/op/erosion.hpp>

#include <soillib/op/cu_common.cu>

#include "erosion_thermal.cu"

namespace soil {

//
// Model Geometry and Lookup Procedures
//

__device__ int __nearest(const model_t& model, const vec2 pos){

  return model.index.flatten(pos);

}

__device__ vec3 __normal(const model_t& model, const vec2 pos, const vec3 scale){

  lerp5_t<float> lerp;
  lerp.gather(model.height, model.sediment, model.index, ivec2(pos));
  const vec2 grad = lerp.grad(scale);
  return glm::normalize(vec3(-grad.x, -grad.y, 1.0f));

}

__device__ float __slope(const model_t& model, const param_t& param, const vec2 pos, const vec2 dir) {

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  if(glm::length(dir) == 0.0f){
    return 0.0f;
  }

  const vec2 npos = pos + glm::normalize(dir);
  if(npos.x < 0.5f) return -param.exitSlope;
  if(npos.y < 0.5f) return -param.exitSlope;
  if(model.index.oob(npos)){
    return -param.exitSlope;
  }
  
  const int ind = model.index.flatten(pos);
  const int nind = model.index.flatten(npos);
  float h0 = (model.height[ind] + model.sediment[ind])*scale.z;
  float h1 = (model.height[nind] + model.sediment[nind])*scale.z;
  return (h1 - h0)/glm::length(cl);

}

//
// Derived Quantities
//

__device__ vec2 __avespeed(const model_t& model, const particle_t& part){
  
  const float discharge = model.discharge[part.ind];
  if(discharge == 0.0f)
    return vec2(0.0f);
  
  const vec2 momentum = model.momentum[part.ind];
  return momentum / discharge;

}

//
// Mass-Transfer Functions
//
//  Note that in the transport rates, the value is not clamped, since
//  value-clamping and limiting only occurs in the composite sum.
//

//! Mass Deposition Rate
__device__ float __deposit(const param_t& param, const float mass){

  const float kd = param.depositionRate;  // Fluvial Deposition Rate [1/y]
  const float deposit = kd * mass;        // Deposited Mass [kg]
  return deposit;

}

//! Mass Suspension Rate
__device__ float __suspend(const param_t& param, const vec2 momentum, const float discharge, const float slope, const float vol, const float Area){

  if(discharge < 1.0f)
    return 0.0f;
  
  const float alpha = 0.1333f;
  const float fD = 0.1f;                  //!< Darcy-Weisbach Friction Factor
  const float rho = 1.0f;                 //!< Density of Fluid [kg/m^3]
  const float ks = param.suspensionRate;  // Fluvial Suspension Rate [(m^3/y)^-0.4]
  
  const float velocity = glm::length(momentum / discharge);
  const float shear = 0.125f * fD * rho * velocity * velocity;
  const float power = pow(shear * velocity, alpha); // Stream Power Function
  const float suspend = ks * power * vol;           // Concentration
  
  const float mask = (slope < 0.0f)?1.0f:0.0f;  // Activation Function
  return mask * suspend;                        // [kg/s] (Activated)

}

//! Overall Mass-Transfer Rate
__device__ float __transfer(const model_t& model, const param_t& param, const float mass, const vec2 momentum, const float discharge, const float slope, const float vol){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float deposit = __deposit(param, mass);
  const float suspend = __suspend(param, momentum, discharge, slope, vol, Ac);
  float transfer = (deposit - suspend);
  return transfer;

}

// Note: Maxtransfer here is damped for stability. This should be
//  attempted to be removed using alternative stabilizing methods.
__device__ float __limit(float transfer, const float mass, const float slope, const vec3 scale){

  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  if(transfer <= 0.0f){
    const float maxtransfer = 0.1f * slope * glm::length(cl) / scale.z * Z;
    const float tmin = transfer * glm::min(1.0f, glm::abs(maxtransfer/transfer));
    transfer = glm::max(transfer, tmin);
  }

  transfer = glm::min(transfer, mass);  // Limit by Mass
  return transfer;

}

__global__ void mt_fluvial(model_t model, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= model.height.elem())
    return;

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  const float mass = model.mass[n];                 // Suspended Mass Function
  const float discharge = model.discharge[n];       // Discharge Function
  const vec2 momentum = model.momentum[n];
  const vec2 pos = model.index.unflatten(n);
  const float slope = __slope(model, param, pos + vec2(0.5), momentum); // Local Slope Function

//  const float dt = param.timeStep;        // Geological Timestep [y]
  float transfer = __transfer(model, param, mass, momentum, discharge, slope, discharge);
  transfer = __limit(transfer, mass, slope, scale);

  // Single-Material Mass-Transfer
  model.height[n] += transfer / Z;

  // Multi-Material Mass-Transfer
//  if(transfer >= 0.0f){
//
//    model.sediment[n] += transfer / Z;
//
//  } else {
//
//    const float maxtransfer = model.sediment[n] * Z;
//    float t1 = transfer * glm::min(1.0f, glm::abs(maxtransfer/transfer));
//    model.sediment[n] += t1 / Z;
//
//    transfer -= t1;
//    model.height[n] += transfer / Z;
//
//  }
  
}

//! Fluvial Erosion Mass-Transfer System
//! Single-Material
//!
__device__ void __integrate_mt(model_t& model, const param_t& param, particle_t& part){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  const float ds = glm::length(cl)/glm::length(part.speed); // Dynamic Time-Step
  
  const float mass = part.sed;
  float deposit = param.depositionRate * mass;
  deposit = glm::min(deposit, mass);
  part.sed -= deposit;

}

//
// Core Procedures
// 

//! Position Sampling Procedure
//!
//! Based on a choice of sampling method for the position,
//! we can also compute the probability. Finally, we also
//! determine the index of the closest support point.
__device__ void __sample(particle_t& part, model_t& model, const size_t n, const size_t N){

  part.pos = vec2 {
    curand_uniform(&model.rand[n])*float(model.index[0]),
    curand_uniform(&model.rand[n])*float(model.index[1])
  };
  part.ind = __nearest(model, part.pos);

  const float P = 1.0f / float(model.index.elem()); // Sampling Probability
  part.Q = P * float(N);                            // Sampling Weight

}

//! Track the Differential Quantities along Trajectories
__device__ void __track(model_t& model, const particle_t& part){

  atomicAdd(&model.mass_track[part.ind], (part.sed)/part.Q);
  atomicAdd(&model.discharge_track[part.ind], (part.vol)/part.Q);
  atomicAdd(&model.momentum_track[part.ind].x, (part.vol*part.dspeed.x)/part.Q);
  atomicAdd(&model.momentum_track[part.ind].y, (part.vol*part.dspeed.y)/part.Q);

}

//! Initialize Particle Data from Model
__device__ void __init(particle_t& part, model_t& model, const param_t& param){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float& R = param.rainfall;        // Rainfall Amount  [m/y]
  const float& g = param.gravity;         // Specific Gravity [m/s^2]
  const float& nu = param.viscosity;      // Kinematic Viscosity [m^2/s]

  // Initial Velocity Estimate
  const vec3 normal = __normal(model, part.pos, scale);
  const vec2 average_speed = __avespeed(model, part);
  part.speed = g * vec2(normal.x, normal.y) + nu * average_speed;

  // Initial Tracking Values

  part.vol = Ac * R;        //!< Volume Rate [m^3/s]
  part.dspeed = part.speed; //!< Velocity Rate [m^2/s^2]

  // Initial Sediment Value:
  // Note that there is a maximum amount that can theoretically
  //  be suspended, which is when it is in balance with the amount
  //  that would also be deposited. We can use this to cap the value.
  const float discharge = model.discharge[part.ind];                // Discharge Function
  const float slope = __slope(model, param, part.pos, part.speed);  // Local Slope Function
  const float suspend = __suspend(param, part.speed*discharge, discharge, slope, part.vol, Ac);
  part.sed = suspend;

}

//! Move the Particle along the Trajectory
//!
//! Note: Return's false if we wish to terminate!
//!
__device__ void __move(const model_t& model, particle_t& part){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  const float ds = glm::length(cl)/glm::length(part.speed);

  part.pos = part.pos + ds * (part.speed / cl);
  part.ind = __nearest(model, part.pos);

}

//! Integrate Sub-Solution Quantities in Quasi-Static Time
__device__ void __integrate(const model_t& model, const param_t& param, particle_t& part){
  
  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  
  const float g = param.gravity;          // Specific Gravity [m/s^2]
  const float k1 = param.bedShear;        // Shear-Stress Bed-Shear
  const float k2 = param.viscosity;       // Shear-Stress Viscosity
  
  // Dynamic Time-Step [s]
  const float ds = glm::length(cl)/glm::length(part.speed);
  
  const vec3 normal = __normal(model, part.pos, scale);
  const vec2 average_speed = __avespeed(model, part);
  
  //! Explicit Euler Forward Integration for Gravity
  part.speed = part.speed + ds * g * vec2(normal.x, normal.y);

  //! Implicit Euiler Forward Integration for Bed Shear-Stress and Viscosity
  part.speed =  1.0f/(1.0f + ds * (k1+k2))*part.speed + ds*k2/(1.0f + ds*(k1+k2))*average_speed;
  part.dspeed = 1.0f/(1.0f + ds * (k1+k2))*part.dspeed;

  //! Implicit Euler Forward Integration for Volume Evaporation
  part.vol = 1.0f/(1.0f + ds*param.evapRate)*part.vol;

}

//! Model-Agnostic Solution Implementation
//!
//! This function implements the generic solution procedure,
//! which involves sampling the path-space, initializing the
//! transport systems and integrating them along the paths.
//!
__global__ void solve_fluvial(model_t model, const size_t N, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) 
    return;

  particle_t part;              //!< Data along Trajectory / Per-Particle
  __sample(part, model, n, N);  //!< Sample the Trajectory
  __init(part, model, param);   //!< Initialze Differential Quantities

  // Iteratively Integrate along Trajectory
  for(int age = 0; age < param.maxage; ++age){

    __track(model, part);         //!< Accumulate Estimate
    __move(model, part);          //!< Move Trajectory
    if(model.index.oob(part.pos))
      break;

    __integrate_mt(model, param, part); //!< Integrate Mass-Transfer
    __integrate(model, param, part);    //!< Integrate Differential Equation

    if(glm::length(part.speed) == 0.0f)
      break;

  }

}

//
// Erosion Function
//

void erode(model_t& model, const param_t param, const size_t steps){

  if(model.height.host() != soil::host_t::GPU){
    throw soil::error::mismatch_host(soil::host_t::GPU, model.height.host());
  }

  if(model.discharge.host() != soil::host_t::GPU){
    throw soil::error::mismatch_host(soil::host_t::GPU, model.discharge.host());
  }

  if(model.momentum.host() != soil::host_t::GPU){
    throw soil::error::mismatch_host(soil::host_t::GPU, model.momentum.host());
  }
  
  //
  // Initialize Rand-State Buffer (One Per Sample)
  //

  const size_t n_samples = param.samples;

  // note: the offset in the sequence should be number of times rand is sampled
  // that way the sampling procedure becomes deterministic

  if(model.rand.elem() != n_samples){
    model.rand = soil::buffer_t<curandState>(n_samples, soil::host_t::GPU);
    seed(model.rand, 0, 2 * model.age);
  }
  
  // Allocate Estimate Buffers for Transported Quantities
  model.mass_track = soil::buffer_t<float>(model.mass.elem(), soil::host_t::GPU);
  model.discharge_track = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);
  model.momentum_track = soil::buffer_t<vec2>(model.momentum.elem(), soil::host_t::GPU);
  model.debris_track = soil::buffer_t<float>(model.debris.elem(), soil::host_t::GPU);

  //
  // Execute Solution
  //

  for(size_t step = 0; step < steps; ++step){

    // Reset Estimates
    set(model.discharge_track, 0.0f);
    set(model.momentum_track, vec2(0.0f));
    set(model.mass_track, 0.0f);
    set(model.debris_track, 0.0f);
    cudaDeviceSynchronize();

    // Solve Estimates
    solve_fluvial<<<block(n_samples, 512), 512>>>(model, n_samples, param);
    solve_debris<<<block(n_samples, 512), 512>>>(model, n_samples, param);
    cudaDeviceSynchronize();

    //
    // Debris Flow Kernel
    //

    // Filter Estimates
    filter(model.momentum, model.momentum_track, param.lrate);
    filter(model.discharge, model.discharge_track, param.lrate);
    filter(model.mass, model.mass_track, param.lrate);
    filter(model.debris, model.debris_track, param.lrate);
    cudaDeviceSynchronize();

    // Execute Height-Map Mass-Transfer
    mt_fluvial<<<block(model.height.elem(), 1024), 1024>>>(model, param);
    mt_debris<<<block(model.height.elem(), 1024), 1024>>>(model, param);

    // Increment Model Age for Rand-State Initialization
    model.age++;

  }

}

} // end of namespace soil

#endif