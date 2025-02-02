#ifndef SOILLIB_NODE_EROSION_CU
#define SOILLIB_NODE_EROSION_CU
#define HAS_CUDA

#include <soillib/node/erosion.hpp>
#include <soillib/util/error.hpp>
#include <soillib/node/lerp.cu>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <iostream>

#include "erosion_thermal.cu"

namespace soil {

namespace {

template<typename T>
struct sample_t {
  glm::ivec2 pos;
  T value;
  bool oob = true;
};

template<typename T, typename I>
__device__ void gather(const soil::buffer_t<T> &buffer_t, const I index, glm::ivec2 p, sample_t<T> px[5], sample_t<T> py[5]) {
  for (int i = 0; i < 5; ++i) {

    const glm::ivec2 pos_x = p + glm::ivec2(-2 + i, 0);
    if (!index.oob(pos_x)) {
      px[i].oob = false;
      px[i].pos = pos_x;

      const size_t ind = index.flatten(pos_x);
      px[i].value = buffer_t[ind];
    }

    const glm::ivec2 pos_y = p + glm::ivec2(0, -2 + i);
    if (!index.oob(pos_y)) {
      py[i].oob = false;
      py[i].pos = pos_y;

      const size_t ind = index.flatten(pos_y);
      py[i].value = buffer_t[ind];
    }
  }
}

template<std::floating_point T>
__device__ glm::vec2 gradient_detailed(sample_t<T> px[5], sample_t<T> py[5]) {

  glm::vec2 g = glm::vec2(0, 0);

  // X-Element
  if (!px[0].oob && !px[4].oob)
    g.x = (1.0f * px[0].value - 8.0f * px[1].value + 8.0f * px[3].value - 1.0f * px[4].value) / 12.0f;

  else if (!px[0].oob && !px[3].oob)
    g.x = (1.0f * px[0].value - 6.0f * px[1].value + 3.0f * px[2].value + 2.0f * px[3].value) / 6.0f;

  else if (!px[0].oob && !px[2].oob)
    g.x = (1.0f * px[0].value - 4.0f * px[1].value + 3.0f * px[2].value) / 2.0f;

  else if (!px[1].oob && !px[4].oob)
    g.x = (-2.0f * px[1].value - 3.0f * px[2].value + 6.0f * px[3].value - 1.0f * px[4].value) / 6.0f;

  else if (!px[2].oob && !px[4].oob)
    g.x = (-3.0f * px[2].value + 4.0f * px[3].value - 1.0f * px[4].value) / 2.0f;

  else if (!px[1].oob && !px[3].oob)
    g.x = (-1.0f * px[1].value + 1.0f * px[3].value) / 2.0f;

  else if (!px[2].oob && !px[3].oob)
    g.x = (-1.0f * px[2].value + 1.0f * px[3].value) / 1.0f;

  else if (!px[1].oob && !px[2].oob)
    g.x = (-1.0f * px[1].value + 1.0f * px[2].value) / 1.0f;

  // Y-Element

  if (!py[0].oob && !py[4].oob)
    g.y = (1.0f * py[0].value - 8.0f * py[1].value + 8.0f * py[3].value - 1.0f * py[4].value) / 12.0f;

  else if (!py[0].oob && !py[3].oob)
    g.y = (1.0f * py[0].value - 6.0f * py[1].value + 3.0f * py[2].value + 2.0f * py[3].value) / 6.0f;

  else if (!py[0].oob && !py[2].oob)
    g.y = (1.0f * py[0].value - 4.0f * py[1].value + 3.0f * py[2].value) / 2.0f;

  else if (!py[1].oob && !py[4].oob)
    g.y = (-2.0f * py[1].value - 3.0f * py[2].value + 6.0f * py[3].value - 1.0f * py[4].value) / 6.0f;

  else if (!py[2].oob && !py[4].oob)
    g.y = (-3.0f * py[2].value + 4.0f * py[3].value - 1.0f * py[4].value) / 2.0f;

  else if (!py[1].oob && !py[3].oob)
    g.y = (-1.0f * py[1].value + 1.0f * py[3].value) / 2.0f;

  else if (!py[2].oob && !py[3].oob)
    g.y = (-1.0f * py[2].value + 1.0f * py[3].value) / 1.0f;

  else if (!py[1].oob && !py[2].oob)
    g.y = (-1.0f * py[1].value + 1.0f * py[2].value) / 1.0f;

  return g;
}

__device__ vec2 gradient(const model_t& model, const vec2 pos){

  sample_t<float> px[5], py[5];
  gather<float, soil::flat_t<2>>(model.height, model.index, ivec2(pos), px, py);
  return gradient_detailed<float>(px, py);

}

__device__ float sigmoid(float x) {
  return x / sqrt(1.0f + x*x);
}

int block(const int elem, const int thread){
  return (elem + thread - 1)/thread;
}

}

//
// Randstate and Estimate Initialization / Filtering
//

__global__ void init_randstate(curandState* states, const size_t N, const size_t seed, const size_t offset) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;
  
  curand_init(seed, n, 2*offset, &states[n]); // scale by 2 because we take two random samples per iteration

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

  model.suspended[n] = mix<float>(model.suspended[n], model.suspended_track[n], 0.9f);
  model.equilibrium[n] = mix<float>(model.equilibrium[n], model.equilibrium_track[n], 0.9f);

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

  float vol = 1.0f;
  float sed = 0.0f;

  const vec2 grad = gradient(model, pos);
  const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
  const vec2 average_speed = (model.momentum[find]) / (1.0f + model.discharge[find]);
  vec2 speed = g * vec2(normal.x, normal.y) + (mu / vol) * average_speed;

  vec2 dspeed = speed;

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

    vec2 nspeed = speed;
    vec2 npos = pos;

    // Viscosity Contribution

    const vec2 average_speed = (model.momentum[find] + vol * speed) / (1.0f + model.discharge[find] + vol);
    nspeed += mu * (average_speed - speed);

    // Gravity Contribution

    const vec2 grad = gradient(model, pos);
    const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
    nspeed += g * vec2(normal.x, normal.y);

    //
    // Time-Step Normalization
    //

    // Note: Here we should see if we can use the length of the speed
    //  vector as the inverse of the time-step. That would help scale
    //  things correctly.
    //  Additionally, we should limit the magnitude of the velocity,
    //  because it does have the change to run-away despite the viscosity.

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
    // Mass-Transfer
    //

    const float equilibrium = vol * equ_frac(model, pos, npos, param);

    //
    // Accumulate Estimated Values
    //

    // Note: Accumulation Occurds at Current Position

    atomicAdd(&model.discharge_track[find], P*vol);
    atomicAdd(&model.momentum_track[find].x, P*vol*dspeed.x);
    atomicAdd(&model.momentum_track[find].y, P*vol*dspeed.y);

    // Note: Both of these work but are slightly different. Find out why!
    
    //atomicAdd(&model.equilibrium_track[find], equilibrium);
    //atomicAdd(&model.suspended_track[find], sed);
    atomicAdd(&model.height[find], -k*(equilibrium - sed));

    //
    // Integrate Sub-Solution Quantities
    //

    vol *= (1.0f - param.evapRate);
    dspeed += - (mu / vol)*dspeed;
    sed += k * (equilibrium - sed);

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

__global__ void apply_height(model_t model, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= model.elem) return;

  // what is the correct way to scale this difference value?
  // since equilibrium is effectively weighted by vol,
  // and sediment is scaled by volume as well in theory...
  // so one is an average concentration, the other is an
  // an average equilibrium concentration.

  const float k = glm::clamp(param.depositionRate, 0.0f, 1.0f);
  const float equilibrium = model.equilibrium[n];
  const float discharge = model.discharge[n];
  const float sediment = model.suspended[n];
  if(discharge > 0.0f){
    model.height[n] += -param.hscale*k*(equilibrium - sediment)/(discharge);
  }

}

//
// Erosion Function
//

void gpu_erode(model_t& model, const param_t param, const size_t steps, const size_t n_samples){

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

  // note: the offset in the sequence should be number of times rand is sampled
  // that way the sampling procedure becomes deterministic
  curandState* randStates;
  cudaMalloc((void**)&randStates, n_samples * sizeof(curandState));
  init_randstate<<<block(n_samples, 512), 512>>>(randStates, n_samples, 0, model.age);
  cudaDeviceSynchronize();

  //
  // Estimate Buffers
  //

  // Note: Extract this Allocation
  auto buf_discharge = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);
  auto buf_suspended = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);
  auto buf_momentum = soil::buffer_t<vec2>(model.discharge.elem(), soil::host_t::GPU);
  model.discharge_track = buf_discharge;
  model.suspended_track = buf_suspended;
  model.momentum_track = buf_momentum;

  auto buf_equilibrium = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);
  auto buf_equilibrium_track = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);
  model.equilibrium = buf_equilibrium;
  model.equilibrium_track = buf_equilibrium_track;

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
    // Apply Height-Map Updates
    //

    // apply the suspension difference...
    // apply_height<<<block(model.elem, 1024), 1024>>>(model, param);

    // atomic add operations might still be coming in -
    // we have to be done before cascading or this fails...
    // we can't be computing the differences before they are determined...

    compute_cascade<<<block(model.elem, 1024), 1024>>>(model, model.discharge_track, param);
    apply_cascade<<<block(model.elem, 1024), 1024>>>(model, model.discharge_track, param);
    cudaDeviceSynchronize();

    // Increment Model Age for Rand-State Initialization
    model.age++;

  }

  cudaFree(randStates);

}

} // end of namespace soil

#endif