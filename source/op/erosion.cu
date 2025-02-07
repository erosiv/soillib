#ifndef SOILLIB_NODE_EROSION_CU
#define SOILLIB_NODE_EROSION_CU
#define HAS_CUDA

#include <soillib/util/error.hpp>

#include <cuda_runtime.h>
#include <curand_kernel.h>
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

__global__ void init_randstate(curandState* states, const size_t N, const size_t seed, const size_t offset) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;
  
  curand_init(seed, n, 2*offset, &states[n]); // scale by 2 because we take two random samples per iteration

}

__device__ vec2 gradient(const model_t& model, const vec2 pos){

  lerp5_t<float> lerp;
  lerp.gather(model.height, model.index, ivec2(pos));
  return lerp.grad();

}

__global__ void reset(model_t model){
  
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= model.elem) return;
  
  // Reset Estimation Buffers

  model.discharge_track[n] = 0.0f;
  model.suspended_track[n] = 0.0f;
  model.momentum_track[n] = vec2(0.0f);
  model.equilibrium_track[n] = 0.0f;

}

template<typename T>
__device__ T mix(T a, T b, float w){
  return (1.0f-w)*a + w*b;
}

__global__ void filter(model_t model, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= model.elem) return;

  // Apply Simple Exponential Filter to Noisy Estimates

  model.discharge[n] = mix<float>(model.discharge[n], model.discharge_track[n], param.lrate);
  model.momentum[n] = mix<vec2>(model.momentum[n], model.momentum_track[n], param.lrate);
  model.suspended[n] = mix<float>(model.suspended[n], model.suspended_track[n], param.lrate);

}

//
// Erosion Kernels
//

__device__ float equ_frac(const model_t& model, vec2 pos, vec2 npos, const param_t param){

  const int find = model.index.flatten(pos);
  const int nind = model.index.flatten(npos);

  float h0 = model.height[find];
  float h1 = h0 - param.exitSlope; 
  if(!model.index.oob(npos)){
    h1 = model.height[nind];
  }

  const float discharge = glm::max(0.0f, model.discharge[find]);  // Discharge Volume
  const float slope = (h0 - h1);                  // Local Slope

  return glm::max(slope, 0.0f) * param.entrainment * log(1.0f + discharge);

}

__global__ void solve(model_t model, curandState* randStates, const size_t N, const param_t param){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= N) return;

  //
  // Parameters
  // Note: Scale-Normalize Values
  //

  const float mu = param.momentumTransfer;
  const float g = param.gravity;
  const float k = param.depositionRate;

  //
  // Initial Condition
  //
  
  // Trajectory and Integration State

  const float P = float(model.elem)/float(N); // Sample Probability
  curandState* randState = &randStates[ind];
  vec2 pos = vec2{
    curand_uniform(randState)*float(model.index[0]),
    curand_uniform(randState)*float(model.index[1])
  };

  int find = model.index.flatten(pos);

  // Water and Sediment Masses

  const float rho_vol = 1.0f;
  float vol = 1.0f;
  float mass = rho_vol*vol;

  const vec2 grad = gradient(model, pos);
  const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
  const vec2 average_speed = (model.momentum[find]) / (1.0f + rho_vol*model.discharge[find]);
  vec2 speed = g * vec2(normal.x, normal.y) + (mu / mass) * average_speed;
  vec2 dspeed = speed;

  // Sediment Transport:
  // Doing this differentially is unstable for some reason.
  // For now, we just integrate the solution directly.

//  vec2 npos = pos + sqrt(2.0f)*glm::normalize(speed);
//  const float equilibrium = vol * (equ_frac(model, pos, npos, param));
//  float dsed = k*equilibrium;
//  atomicAdd(&model.height[find], -dsed);
  float sed = 0.0f;

  // Solution Loop:
  //  Solve Conservation Law along Characteristic
  //  Generated by the Flow.

  for(size_t age = 0; age < param.maxage; ++age){

    // Termination Conditions

    if(model.index.oob(pos))      return;
    if(vol < param.minVol)        return;
    if(glm::length(speed) < 1E-4) return;

    //
    // Execute Integration
    //

    // Flow Integration / Trajectory

    const vec2 grad = gradient(model, pos);
    const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
    const vec2 average_speed = (model.momentum[find]) / (1.0f + rho_vol*model.discharge[find]);

    vec2 nspeed = speed + g * vec2(normal.x, normal.y) + (mu / mass)*(average_speed - speed);

    //
    // Time-Step Normalization
    //

    // Note: Here we should see if we can use the length of the speed
    //  vector as the inverse of the time-step. That would help scale
    //  things correctly.
    //  Additionally, we should limit the magnitude of the velocity,
    //  because it does have the change to run-away despite the viscosity.

    vec2 npos = pos;
    if(glm::length(nspeed) > 0.0){
      npos += sqrt(2.0f)*glm::normalize(nspeed);
    } else {
      // note: if the position becomes the same,
      // slope will also be zero
      // meaning equilibrium drops to zero
      // which could cause a chain reaction of deposition
      break;
    }

    //
    // Accumulate Estimated Values
    //

    // Note: Accumulation Occurds at Current Position

    atomicAdd(&model.discharge_track[find], P*vol);
    atomicAdd(&model.momentum_track[find].x, P*mass*dspeed.x);
    atomicAdd(&model.momentum_track[find].y, P*mass*dspeed.y);

    //
    // Mass-Transfer
    //

    // Note: Differential Integration is Unstable due to Height-Map
    // So we do direct integration instead, assuming segmented
    // fluid packets.

    // atomicAdd(&model.suspended_track[find], P*dsed);
    // const float equilibrium = vol * (equ_frac(model, pos, npos, param));
    // atomicAdd(&model.height[find], -k*(equilibrium - model.suspended[find]));
    // dsed *= (1.0f - k);

    // Note: This expression for the equilibrium concentration depends solely on
    // the discharge volume. That means that the small gradient along curved flows
    // leads a meander to progress. To increase this effect, and get the meanders
    // to "buckle", the equilibrium concentration needs to be increased further
    // on the outer edge of the curve.

    const float equilibrium = vol * (equ_frac(model, pos, npos, param));
    atomicAdd(&model.height[find], -k*(equilibrium - sed));
    sed += k * (equilibrium - sed);
    
    // Note: We should limit it so that this doesn't cause a runaway deposition.
    // That occurs when the amount removed is larger than making the flow flat.
    // The particles coming from behind will then hit that wall and deposity everything.
    // Or we make sure we can't deposit more than equal the amount.
    // Finally, we should add a term which is based on the viscosity, meaning that
    // if the difference between the velocity and the target velocity is larger,
    // we scale the equilibrium value because we have a higher shear-stress.

    //
    // Integrate Sub-Solution Quantities
    //

    vol *= (1.0f - param.evapRate);
    dspeed += - (mu / mass)*dspeed;
    mass = rho_vol*vol;

    // Update Position at next Position?
    // We do this because technically,
    // we have moved forward to where
    // the velocity has changed as specified.

    //
    // Update Trajectory
    //

    pos = npos;
    speed = nspeed;
    find = model.index.flatten(pos);

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
  curandState* randStates;
  cudaMalloc((void**)&randStates, n_samples * sizeof(curandState));
  init_randstate<<<block(n_samples, 512), 512>>>(randStates, n_samples, 0, model.age);
  cudaDeviceSynchronize();

  //
  // Estimate Buffers
  //

  model.discharge_track = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);
  model.suspended_track = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);
  model.momentum_track = soil::buffer_t<vec2>(model.discharge.elem(), soil::host_t::GPU);
  model.equilibrium = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);
  model.equilibrium_track = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);

  //
  // Execute Solution
  //

  for(size_t step = 0; step < steps; ++step){

    //
    // Reset, Solve, Filter, Apply
    //

    reset<<<block(model.elem, 1024), 1024>>>(model);
    cudaDeviceSynchronize();

    solve<<<block(n_samples, 512), 512>>>(model, randStates, n_samples, param);
    cudaDeviceSynchronize();
 
    filter<<<block(model.elem, 1024), 1024>>>(model, param);
    cudaDeviceSynchronize();

    //
    // Thermal Erosion Kernel
    //

    compute_cascade<<<block(model.elem, 1024), 1024>>>(model, model.discharge_track, param);
    apply_cascade<<<block(model.elem, 1024), 1024>>>(model, model.discharge_track, param);
    cudaDeviceSynchronize();

   
    model.age++; // Increment Model Age for Rand-State Initialization

  }

  cudaFree(randStates);

}

} // end of namespace soil

#endif