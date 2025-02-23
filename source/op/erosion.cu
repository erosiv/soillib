#ifndef SOILLIB_NODE_EROSION_CU
#define SOILLIB_NODE_EROSION_CU
#define HAS_CUDA

#include <soillib/util/error.hpp>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <iostream>

#include <soillib/op/common.hpp>
#include <soillib/op/gather.hpp>
#include <soillib/op/erosion.hpp>

#include "erosion_thermal.cu"

namespace soil {

//
// Randstate and Estimate Initialization / Filtering
//

__global__ void seed(buffer_t<curandState> buffer, const size_t seed, const size_t offset) {
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= buffer.elem()) return;
  curand_init(seed, n, offset, &buffer[n]);
}

__global__ void reset(model_t model){
  
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= model.elem) return;
  
  // Reset Estimation Buffers

  model.discharge_track[n] = 0.0f;
  model.momentum_track[n] = vec2(0.0f);

}

__global__ void filter(model_t model, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= model.elem) return;

  // Apply Simple Exponential Filter to Noisy Estimates

  model.discharge[n] = glm::mix(model.discharge[n], model.discharge_track[n], param.lrate);
  model.momentum[n] = glm::mix(model.momentum[n], model.momentum_track[n], param.lrate);

}

//
// Erosion Kernels
//

__global__ void solve(model_t model, const size_t N, const param_t param){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= N) return;

  //
  // Parameters
  //

  // Scaled Domain Parameters
  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m]
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float R = param.rainfall;         // Rainfall Amount  [m/y]
  const float g = param.gravity;          // Specific Gravity [m/s^2]
  const float kd = param.depositionRate;  // Fluvial Deposition Rate

  // Two Problem Parameters:
  const float nu = param.viscosity;// * 24000.0f;      // Kinematic Viscosity [m^2/s]
  const float ks = kd * param.entrainment * 7E-7f;  // Fluvial Suspension Rate

  //
  // Position Sampling Procedure:
  //  Note: If we isolate this, we also wish to return the probability
  //  that any individual sample was chosen. For now, it is uniform.
  //  Additionally, this can be area based, but ultimately depends
  //  on the actual implementation of the sampling procedure.
  //

  curandState* randState = &model.rand[ind];
  vec2 pos = vec2{
    curand_uniform(randState)*float(model.index[0]),
    curand_uniform(randState)*float(model.index[1])
  };
  const float P = 1.0f / float(model.index.elem());
  int find = model.index.flatten(pos);

  //
  // Transport Initial Condition
  //

  float dt = 1.0f;    // [y]
  float vol = Ac * R; // [m^3/y]
  float sed = 0.0f;

  //
  // Trajectory Initial Condition
  //

  // Surface Normal Vector
  lerp5_t<float> lerp;
  lerp.gather(model.height, model.sediment, model.index, ivec2(pos));
  const vec2 grad = lerp.grad(scale);
  const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));

  // Average Local Velocity
  const vec2 momentum = model.momentum[find];
  const float discharge = model.discharge[find];
  vec2 average_speed = vec2(0.0f);
  if(discharge > 0.0f) {
    average_speed = momentum / discharge;
  }

  // Initial Velocity Estimate
  vec2 speed = g * vec2(normal.x, normal.y) + nu * average_speed;
  if(glm::length(speed) == 0.0f)
    return;

  float ds = glm::length(cl) / glm::length(speed);
  vec2 npos = pos + ds*(speed / cl);
  vec2 dspeed = speed;

  // Solution Loop:
  //  Solve Conservation Law along Characteristic
  //  Generated by the Flow.

  for(size_t age = 0; age < param.maxage; ++age){

    //
    // Accumulate Estimated Values
    //

    // Note: Accumulation Occurds at Current Position

    atomicAdd(&model.discharge_track[find], (1.0f/P/N)*vol);
    atomicAdd(&model.momentum_track[find].x, (1.0f/P/N)*vol*dspeed.x);
    atomicAdd(&model.momentum_track[find].y, (1.0f/P/N)*vol*dspeed.y);

    //
    // Mass-Transfer
    //  Compute Equilibrium Mass from Slope and Discharge
    //  Transfer Mass and Scale by Sampling Probability
    
    float discharge = model.discharge[find];
    float slope = -param.exitSlope;
    if(!model.index.oob(npos)){
      const int nind = model.index.flatten(npos);
      float h0 = (model.height[find] + model.sediment[find])*scale.z;
      float h1 = (model.height[nind] + model.sediment[nind])*scale.z;
      slope = (h1 - h0)/glm::length(cl);
    }

    float deposit = kd * sed;
    float suspend = ks * vol  * glm::max(0.0f, -slope) * pow(discharge, 0.4f);
    float transfer = (deposit - suspend);

    // Erosion Stability: Limit Transfer by Slope
    //  Note: This is a hard-max function applied to an explicit euler scheme.
    //  This combination can be replaced by an implicit scheme on its own.

    if(transfer > 0.0f){ // Add Material to Map
      if(slope > 0.0f){
        const float maxtransfer = slope * glm::length(cl)/scale.z;
        transfer = glm::min(maxtransfer, transfer / P / float(N)) * P * float(N);
      }
    } 
    
    else if(transfer < 0.0f) { // Remove Material from Map
      if(slope < 0.0f){
        const float maxtransfer = -slope * glm::length(cl)/scale.z;
        transfer = -glm::min(maxtransfer, -transfer / P / float(N)) * P * float(N);
      }
    }

    // Simple Equilibrium, Single-Material Mass-Transfer
    
//    if(transfer > 0.0f){
//      transfer = glm::min(sed, transfer);
//    }
//
//    atomicAdd(&model.height[find], transfer / P / float(N));
//    sed -= transfer;

    // Simple Equilibrium, Multi-Material Mass Transfer

    if(transfer > 0.0f){      // Add Material to Map

      transfer = glm::min(sed, transfer);
      atomicAdd(&model.sediment[find], transfer / P / float(N));
      sed -= transfer;

    }

    else if(transfer < 0.0f){ // Remove Sediment from Map

      const float maxtransfer = model.sediment[find];
      float t1 = -glm::min(maxtransfer, -transfer / P / float(N)) * P * float(N);

      atomicAdd(&model.sediment[find], t1/ P / float(N));
      sed -= t1;

      transfer -= t1;
      atomicAdd(&model.height[find], transfer / P / float(N));
      sed -= transfer;

    }

    //
    // Integrate Sub-Solution Quantities
    //  Note: Integrated in Quasi-Static Time
    //    using an Implicit Forward Scheme

    vol = 1.0f/(1.0f + ds*param.evapRate)*vol;
    dspeed = 1.0f/(1.0f + ds*nu)*dspeed;

    //
    // Flow Integration / Trajectory
    //

    pos = npos;
    if(model.index.oob(pos))
      break;

    find = model.index.flatten(pos);

    lerp5_t<float> lerp;
    lerp.gather(model.height, model.sediment, model.index, ivec2(pos));
    const vec2 grad = lerp.grad(scale);
    const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));

    discharge = model.discharge[find];
    const vec2 momentum = model.momentum[find];
    vec2 average_speed = vec2(0.0f);
    if(discharge > 0.0f){
      average_speed = momentum / discharge;
    }

    // Implicit Euler Forward Integration:

    speed = speed + ds * g * vec2(normal.x, normal.y);
    speed = 1.0f/(1.0f + ds*nu)*speed + ds*nu/(1.0f + ds*nu)*average_speed;

    if(glm::length(speed) == 0.0f)
      break;

    ds = glm::length(cl)/glm::length(speed);
    npos = pos + ds * (speed / cl);

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
    seed<<<block(n_samples, 512), 512>>>(model.rand, 0, 2 * model.age);
    cudaDeviceSynchronize();
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

    reset<<<block(model.elem, 1024), 1024>>>(model);
    cudaDeviceSynchronize();

    solve<<<block(n_samples, 512), 512>>>(model, n_samples, param);
    cudaDeviceSynchronize();
 
    filter<<<block(model.elem, 1024), 1024>>>(model, param);
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