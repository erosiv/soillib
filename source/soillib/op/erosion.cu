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
// Local Particle State
//

struct particle_t {

  vec2 pos;
  float P;
  int ind;

  vec2 npos;
  vec2 speed;
  vec2 dspeed;
  float ds;

  float vol;
  float sed;

};

//
// Model Geometry and Lookup Procedures
//

__device__ vec3 __normal(const model_t& model, const vec2 pos, const vec3 scale){

  lerp5_t<float> lerp;
  lerp.gather(model.height, model.sediment, model.index, ivec2(pos));
  const vec2 grad = lerp.grad(scale);
  return glm::normalize(vec3(-grad.x, -grad.y, 1.0f));

}

__device__ int __nearest(const model_t& model, const vec2 pos){
  return model.index.flatten(pos);
}

__device__ vec2 __avespeed(const model_t& model, const particle_t& part){
  const vec2 momentum = model.momentum[part.ind];
  const float discharge = model.discharge[part.ind];
  if(discharge == 0.0f)
    return vec2(0.0f);
  return momentum / discharge;
}

//
// Position Sampling Procedure:
//  Choose a sampling method, determine the sampling probability,
//  determine the contribution weight from the sample count.
//  Then get the sample and determine the nearest point.
//

//! Position Sampling Procedure
//!
//! Based on a choice of sampling method for the position,
//! we can also compute the probability. Finally, we also
//! determine the index of the closest support point.
__device__ void __sample(particle_t& part, const model_t& model, const size_t n){

  part.pos = __sample_2D(&model.rand[n], model.index);
  part.P = 1.0f / float(model.index.elem()); // Sampling Probability
  
  part.ind = __nearest(model, part.pos);

}

//! Initialize Particle Data from Model
//!
__device__ void __init(particle_t& part, const model_t& model, const param_t& param){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float R = param.rainfall;         // Rainfall Amount  [m/y]
  const float g = param.gravity;          // Specific Gravity [m/s^2]
  const float nu = param.viscosity;       // Kinematic Viscosity [m^2/s]

  // Surface Normal Vector
  const vec3 normal = __normal(model, part.pos, scale);
  const vec2 average_speed = __avespeed(model, part);

  // Initial Velocity Estimate
  part.speed = g * vec2(normal.x, normal.y) + nu * average_speed;
  if(glm::length(part.speed) == 0.0f)
    return;

  part.ds = glm::length(cl) / glm::length(part.speed);
  part.npos = part.pos + part.ds*(part.speed / cl);
  part.dspeed = part.speed;

  part.vol = Ac * R; // [m^3/y]
  part.sed = 0.0f;

}

//! Note: Return's false if we wish to terminate!
__device__ bool __move(const model_t& model, particle_t& part){

  part.pos = part.npos;
  if(model.index.oob(part.pos))
    return false;
    part.ind = __nearest(model, part.pos);
  return true;

}

__device__ void __track(model_t& model, const particle_t& part, const size_t N){

  const float Q = part.P * float(N);

  atomicAdd(&model.discharge_track[part.ind], (1.0f/Q)*part.vol);
  atomicAdd(&model.momentum_track[part.ind].x, (1.0f/Q)*part.vol*part.dspeed.x);
  atomicAdd(&model.momentum_track[part.ind].y, (1.0f/Q)*part.vol*part.dspeed.y);

}

__device__ bool __integrate(const model_t& model, particle_t& part, const param_t& param){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float g = param.gravity;          // Specific Gravity [m/s^2]

  //
  // Integrate Sub-Solution Quantities
  //  Note: Integrated in Quasi-Static Time
  //    using an Implicit Forward Scheme

  part.vol = 1.0f/(1.0f + part.ds*param.evapRate)*part.vol;

  // Implicit Euler Forward Integration for Gravity
  // 
  // Shear-Stress and Viscosity Terms:
  //  The viscosity and shear-stress are some combination
  //  of bed shear-stress from friction, and mixing with the
  //  bulk stream from viscosity. These two terms are linearly
  //  related to the velocity and bulk velocity deviation
  //  by some constants k1 and k2 respectively.
  //
  //  The only key question is how these parameters scale with
  //  the space and time resolution of the simulation.
  
  const float k1 = param.bedShear;
  const float k2 = param.viscosity;

  const vec3 normal = __normal(model, part.pos, scale);
  const vec2 average_speed = __avespeed(model, part);

  part.speed = part.speed + part.ds * g * vec2(normal.x, normal.y);

  part.speed =  1.0f/(1.0f + part.ds * (k1+k2))*part.speed + part.ds*k2/(1.0f + part.ds*(k1+k2))*average_speed;
  part.dspeed = 1.0f/(1.0f + part.ds * (k1+k2))*part.dspeed;

  if(glm::length(part.speed) == 0.0f)
    return false;

  part.ds = glm::length(cl)/glm::length(part.speed);
  part.npos = part.pos + part.ds * (part.speed / cl);

  return true;

}

__device__ float __slope(const model_t& model, const particle_t& part, const param_t& param){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  float slope = -param.exitSlope;
  float h0 = (model.height[part.ind] + model.sediment[part.ind])*scale.z;
  float h1 = h0 + slope * glm::length(cl);
  
  if(!model.index.oob(part.npos)){
    const int nind = model.index.flatten(part.npos);
    h1 = (model.height[nind] + model.sediment[nind])*scale.z;
    slope = (h1 - h0)/glm::length(cl);
  }

  return slope;

}

 __device__ void __masstransfer1(model_t& model, particle_t& part, const param_t& param, const size_t N){

    const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
    const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
    const float Ac = scale.x*scale.y;       // Cell Area [m^2]
    const float Z = Ac * scale.z;           // Height Conversion [m^3]

    const float dt = param.timeStep;        // Geological Timestep [y]
    const float kd = param.depositionRate;  // Fluvial Deposition Rate [1/y]
    const float ks = param.suspensionRate;  // Fluvial Suspension Rate [(m^3/y)^-0.4]

    const float Q = part.P * float(N);

    float discharge = model.discharge[part.ind];  // Discharge Function
    float slope = __slope(model, part, param);    // Slope Function
    float alpha = (slope < 0.0f)?1.0f:0.0f;       // Activation Function
    float suspend = dt * ks * part.vol * slope * alpha * pow(discharge, 0.4f); // [kg]
    float deposit = dt * kd * part.sed;                                        // [kg]

    // Single Material, Implicit Euler Scheme
    //  This use an activation function which lowers the amount transferred
    //  which scales with the amount of equilibriation force. Note that this
    //  tends to over-damp, which is why we don't use it.

//    float kq = ks * part.vol * alpha * pow(discharge, 0.4f) / glm::length(cl);
//    float transfer = 1.0f / (1.0f + dt * kq) * (suspend + deposit);
//    atomicAdd(&model.height[part.ind], transfer / Z / Q);
//    part.sed -= transfer;

    // Single Material, Explicit Euler Scheme
    //  This use an activation function (maxtransfer), which limits the
    //  total amount of mass that can be moved based on the slope.
    //  Similar to the implicit scheme, which uses a similar construction
    //  but that scales with the rate.
  
    // Note: Maxtransfer here is damped for stability. This should be
    //  attempted to be removed using alternative stabilizing methods.
    float transfer = (deposit + suspend);
    const float maxtransfer = 0.1f * slope * glm::length(cl) / scale.z * Z * Q;
    const float tmin = transfer * glm::min(1.0f, glm::abs(maxtransfer/transfer));
    const float tmax = part.sed;
    transfer = glm::clamp(transfer, tmin, tmax);

    atomicAdd(&model.height[part.ind], transfer / Z / Q);
    part.sed -= transfer;

}

__device__ void __masstransfer2(model_t& model, particle_t& part, const param_t& param, const size_t N){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  const float dt = param.timeStep;        // Geological Timestep [y]
  const float kd = param.depositionRate;  // Fluvial Deposition Rate [1/y]
  const float ks = param.suspensionRate;  // Fluvial Suspension Rate [(m^3/y)^-0.4]

  const float Q = part.P * float(N);

  //
  // Mass-Transfer
  //  Compute Equilibrium Mass from Slope and Discharge
  //  Transfer Mass and Scale by Sampling Probability
  
  float discharge = model.discharge[part.ind];  // Discharge Function
  float slope = __slope(model, part, param);    // Slope Function
  float alpha = (slope < 0.0f)?1.0f:0.0f;       // Activation Function
  float suspend = dt * ks * part.vol * slope * alpha * pow(discharge, 0.4f); // [kg]
  float deposit = dt * kd * part.sed;                                        // [kg]

  // Multi-Material Mass Transfer
  float transfer = (deposit + suspend);
  const float maxtransfer = 0.1f * slope * glm::length(cl) / scale.z * Z * Q;
  const float tmin = transfer * glm::min(1.0f, glm::abs(maxtransfer/suspend));
  const float tmax = part.sed;
  transfer = glm::clamp(transfer, tmin, tmax);

  if(transfer > 0.0f){  // Add Material to Map (Note: Single Material Model)

    atomicAdd(&model.sediment[part.ind], transfer / Z / Q);
    part.sed -= transfer;

  }

  else if(transfer < 0.0f){ // Remove Sediment from Map

    const float maxtransfer = 0.1f * model.sediment[part.ind] * Z * Q;
    float t1 = transfer * glm::min(1.0f, glm::abs(maxtransfer/transfer));
    atomicAdd(&model.sediment[part.ind], t1 / Z / Q);
    part.sed -= t1;

    transfer -= t1;
    atomicAdd(&model.height[part.ind], transfer / Z / Q);
    part.sed -= transfer;

  }

}

//
// Model-Agnostic Solution Implementation
//

__global__ void solve(model_t model, const size_t N, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;

  //
  // Transport Initial Condition:
  //  The particle stores the data on a per-
  //  sample along the trajectory basis.
  //

  particle_t part;
  __sample(part, model, n);
  __init(part, model, param);
  const float Q = part.P * float(N);

  // Solution Loop:
  //  Solve Conservation Law along Characteristic
  //  Generated by the Flow.

  for(size_t age = 0; age < param.maxage; ++age){

    // Mass-Transfer
    __masstransfer2(model, part, param, N);

    // Accumulate Estimated Quantities
    __track(model, part, N);
    
    // Shift Trajectory
    if(!__move(model, part))
      break;

    // Integrate Differential Quantities
    if(!__integrate(model, part, param))
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

  //
  // Estimate Buffers
  //

  model.discharge_track = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);
  model.momentum_track = soil::buffer_t<vec2>(model.discharge.elem(), soil::host_t::GPU);

  //
  // Execute Solution
  //

  for(size_t step = 0; step < steps; ++step){

    //
    // Reset, Solve, Filter, Apply
    //

    set(model.discharge_track, 0.0f);
    set(model.momentum_track, vec2(0.0f));

    solve<<<block(n_samples, 512), 512>>>(model, n_samples, param);
    cudaDeviceSynchronize();
 
    filter(model.momentum, model.momentum_track, param.lrate);
    filter(model.discharge, model.discharge_track, param.lrate);
    cudaDeviceSynchronize();

    //
    // Debris Flow Kernel
    //

    debris_flow<<<block(n_samples, 512), 512>>>(model, n_samples, param);
    cudaDeviceSynchronize();

    model.age++; // Increment Model Age for Rand-State Initialization

  }

}

} // end of namespace soil

#endif