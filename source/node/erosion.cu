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

}

__global__ void filter(model_t model, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= model.elem) return;

  // Apply Simple Exponential Filter to Noisy Estimates

  const float w = param.lrate;
  model.discharge[n] = (1.0f-w)*model.discharge[n] + w*model.discharge_track[n];
  model.suspended[n] = (1.0f-w)*model.suspended[n] + w*model.suspended_track[n];
  model.momentum[n] = (1.0f-w)*model.momentum[n] + w*model.momentum_track[n];

}

//
// Erosion Kernels
//

__global__ void solve(model_t model, curandState* randStates, const size_t N, const param_t param){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= N) return;

  // Initial Condition

  const float P = float(model.elem)/float(N); // Sample Probability
  curandState* randState = &randStates[ind];
  vec2 pos = vec2{
    curand_uniform(randState)*float(model.index[0]),
    curand_uniform(randState)*float(model.index[1])
  };

  vec2 speed = vec2(0.0f);
  float vol = 1.0f;
  float sed = 0.0f;

  // Solution Loop:
  //  Solve Conservation Law along Characteristic
  //  Generated by the Flow.

  for(size_t age = 0; age < param.maxage; ++age){

    // Termination Conditions

    if(model.index.oob(pos))  return;
    if(vol < param.minVol)    return;

    // Compute Speed Update

    const int find = model.index.flatten(pos);
    
    // Gravity Contribution
    const vec2 grad = gradient(model, pos);
    const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
    speed += param.gravity * vec2(normal.x, normal.y);

    // Viscosity Contribution
    const float mu = param.momentumTransfer;//::clamp(param.momentumTransfer, 0.0f, 1.0f);
    // const vec2 average_speed = (model.momentum[find] + volume * speed) / (model.discharge[find] + volume);
    const vec2 average_speed = (model.momentum[find]) / (model.discharge[find] + vol);
    speed = speed + mu * (average_speed - speed); // Explicit Euler
    //speed = (speed + mu * average_speed)/(1.0f + mu); // Implicit Euler

    // Normalize Time-Step, Increment

    // Update Trajectory

    if(glm::length(speed) > 0.0){
      speed = sqrt(2.0f)*glm::normalize(speed);
    }

    // next position
    const vec2 npos = pos + speed;

    /*
    // speed has to be limited by something...
    particles.spd[ind] = speed;
    if(glm::length(speed) > 0.0){
      particles.pos[ind] += sqrt(2.0f)*glm::normalize(speed);
    }
    const vec2 npos = particles.pos[ind];
    */

    // Update Volume

    vol *= (1.0f - param.evapRate);

    // Mass-Transfer

    float h0 = model.height[find];
    float h1 = h0 - param.exitSlope; 
    if(!model.index.oob(npos)){
      h1 = model.height[model.index.flatten(npos)];
    }

    const float discharge = model.discharge[find];  // Discharge Volume
    const float slope = (h0 - h1);                  // Local Slope

    const float equilibrium = vol * glm::max(slope, 0.0f) * param.entrainment * log(1.0f + discharge);
    const float mass_diff = (equilibrium - sed);

    // Execute Mass-Transfer

    const float k = glm::clamp(param.depositionRate, 0.0f, 1.0f);
//    particles.susp[ind] = k * mass_diff;
    sed += k * mass_diff;

    // Execute the Differential Tracking

    // At the next position though?
    
    if(!model.index.oob(npos)){
      
      const int nind = model.index.flatten(npos);

      atomicAdd(&model.height[find], -k * mass_diff); // note: important that it happens here?

      atomicAdd(&model.discharge_track[nind], P*vol); // note: normalized by scaling by P

      const vec2 m = vol * speed;
      atomicAdd(&model.momentum_track[nind].x, m.x);
      atomicAdd(&model.momentum_track[nind].y, m.y);

//      const float susp = k * mass_diff;
//      atomicAdd(&model.suspended_track[nind], susp);

    }

    // update position
    pos = npos;

  }

}

/*
__global__ void apply_height(soil::buffer_t<float> height, const soil::buffer_t<float> susp, const param_t param){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= height.elem()) return;
  if(ind >= susp.elem()) return;

  // now why the hell would I do this.. it's not scaled correctly...
  // do we have to divide it by the discharge perhaps?
  // that would scale it down...
  // height[ind] += -1.0f * sigmoid(susp[ind] / param.hscale);

}
*/

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

  //
  // Execute Solution
  //

  for(size_t step = 0; step < steps; ++step){

    //
    // Reset, Solve, Filter, Apply
    //

    reset<<<block(model.elem, 1024), 1024>>>(model);
    solve<<<block(n_samples, 512), 512>>>(model, randStates, n_samples, param);
    filter<<<block(model.elem, 1024), 1024>>>(model, param);
    cudaDeviceSynchronize();

    //
    // Apply Height-Map Updates
    //

    // apply the suspension difference...
    //apply_height<<<block(model.elem, 1024), 1024>>>(model.height, model.suspended, param);

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