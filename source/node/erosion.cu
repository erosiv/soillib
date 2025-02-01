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

int block(const int elem, const int thread){
  return (elem + thread - 1)/thread;
}

}

//
// Utility Kernels
//

template<typename T>
__global__ void fill(soil::buffer_t<T> buf, const T val){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= buf.elem()) return;
  buf[index] = val;
}

__global__ void init_randstate(curandState* states, const size_t N, const size_t seed, const size_t offset) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;
  // scale by 2 because we take two random samples per iteration
  curand_init(seed, n, 2*offset, &states[n]);
}

__global__ void spawn(particle_t particles, curandState* randStates, flat_t<2> index){

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= particles.elem) return;

  curandState* randState = &randStates[n];
  particles.pos[n] = vec2{
    curand_uniform(randState)*float(index[0]),
    curand_uniform(randState)*float(index[1])
  };

  particles.spd[n] = vec2(0.0f);
  particles.vol[n] = 1.0f;
  particles.sed[n] = 0.0f;
  particles.susp[n] = 0.0f;

}

__global__ void filter(soil::buffer_t<float> buffer, const soil::buffer_t<float> buffer_track, const float lrate){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= buffer.elem()) return;
  if(ind >= buffer_track.elem()) return;

  float val = buffer[ind];
  float val_track = buffer_track[ind];
  buffer[ind] = val * (1.0f - lrate) +  val_track * lrate;
}

__global__ void filter(soil::buffer_t<vec2> buffer, const soil::buffer_t<vec2> buffer_track, const float lrate){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= buffer.elem()) return;
  if(ind >= buffer_track.elem()) return;

  vec2 val = buffer[ind];
  vec2 val_track = buffer_track[ind];
  buffer[ind] = val * (1.0f - lrate) +  val_track * lrate;
}

template<typename T>
__global__ void normalize(soil::buffer_t<T> out, const T B, const T P){
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= out.elem()) return;
  out[n] = B + P * out[n];
}

__device__ float sigmoid(float x) {
  return x / sqrt(1.0f + x*x);
}

__global__ void apply_height(soil::buffer_t<float> height, const soil::buffer_t<float> susp, const param_t param){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= height.elem()) return;
  if(ind >= susp.elem()) return;

  // now why the hell would I do this.. it's not scaled correctly...
  // do we have to divide it by the discharge perhaps?
  // that would scale it down...
  // height[ind] += -1.0f * sigmoid(susp[ind] / param.hscale);

}

//
// Erosion Kernels
//

__device__ vec2 gradient(const model_t& model, const vec2 pos){

  sample_t<float> px[5], py[5];
  gather<float, soil::flat_t<2>>(model.height, model.index, ivec2(pos), px, py);
  return gradient_detailed<float>(px, py);

}

__global__ void descend(model_t model, particle_t particles, const param_t param){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= particles.elem) return;

  for(size_t age = 0; age < param.maxage; ++age){

    const vec2 pos = particles.pos[ind];
    if(model.index.oob(pos))
      return;

    const float volume = particles.vol[ind];
    if (volume < param.minVol) {
      return;
    }

    // Compute Speed Update

    const int find = model.index.flatten(pos);
    vec2 speed = particles.spd[ind];

    // Gravity Contribution
    const vec2 grad = gradient(model, pos);
    const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
    speed += param.gravity * vec2(normal.x, normal.y);

    // Viscosity Contribution
    const float mu = param.momentumTransfer;//::clamp(param.momentumTransfer, 0.0f, 1.0f);
    // const vec2 average_speed = (model.momentum[find] + volume * speed) / (model.discharge[find] + volume);
    const vec2 average_speed = (model.momentum[find]) / (model.discharge[find] + volume);
    speed = speed + mu * (average_speed - speed); // Explicit Euler
    //speed = (speed + mu * average_speed)/(1.0f + mu); // Implicit Euler

    // Normalize Time-Step, Increment

    // Update Trajectory

    if(glm::length(speed) > 0.0){
      speed = sqrt(2.0f)*glm::normalize(speed);
    }

    particles.spd[ind] = speed; // actual speed
    particles.pos[ind] += speed;
    const vec2 npos = particles.pos[ind];

    /*
    // speed has to be limited by something...
    particles.spd[ind] = speed;
    if(glm::length(speed) > 0.0){
      particles.pos[ind] += sqrt(2.0f)*glm::normalize(speed);
    }
    const vec2 npos = particles.pos[ind];
    */

    // Update Volume

    particles.vol[ind] *= (1.0f - param.evapRate);

    // Mass-Transfer

    float h0 = model.height[find];
    float h1 = h0 - param.exitSlope; 
    if(!model.index.oob(npos)){
      h1 = model.height[model.index.flatten(npos)];
    }

    const float discharge = model.discharge[find];  // Discharge Volume
    const float vol = particles.vol[ind];           // Water Volume
    const float sed = particles.sed[ind];           // Sediment Mass
    const float slope = (h0 - h1);                  // Local Slope

    const float equilibrium = vol * glm::max(slope, 0.0f) * param.entrainment * log(1.0f + discharge);
    const float mass_diff = (equilibrium - sed);

    // Execute Mass-Transfer

    const float k = glm::clamp(param.depositionRate, 0.0f, 1.0f);
    particles.susp[ind] = k * mass_diff;
    particles.sed[ind] += k * mass_diff;

    // Execute the Differential Tracking

    // At the next position though?
    
    if(!model.index.oob(npos)){
      
      const int nind = model.index.flatten(npos);

      atomicAdd(&model.height[find], -k * mass_diff); // note: important that it happens here?

      atomicAdd(&model.discharge_track[nind], particles.vol[ind]);

      const vec2 m = particles.vol[ind] * particles.spd[ind];
      atomicAdd(&model.momentum_track[nind].x, m.x);
      atomicAdd(&model.momentum_track[nind].y, m.y);

      const float susp = k * mass_diff;
      atomicAdd(&model.suspended_track[nind], susp);

    }

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
  // Particle Buffers
  //

  // Note: Extract this Allocation
  auto buf_discharge = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);
  auto buf_suspended = soil::buffer_t<float>(model.discharge.elem(), soil::host_t::GPU);
  auto buf_momentum = soil::buffer_t<vec2>(model.discharge.elem(), soil::host_t::GPU);
  model.discharge_track = buf_discharge;
  model.suspended_track = buf_suspended;
  model.momentum_track = buf_momentum;

  //! \todo remove this allocation, as well as the randstate allocation
  particle_t particles{n_samples};

  //
  // Initialize Rand-State Buffer
  //

  // note: the offset in the sequence should be number of times rand is sampled
  // that way the sampling procedure becomes deterministic
  curandState* randStates;
  cudaMalloc((void**)&randStates, n_samples * sizeof(curandState));
  init_randstate<<<block(n_samples, 512), 512>>>(randStates, n_samples, 0, model.age);

  cudaDeviceSynchronize();

  //
  // Execute Erosion Loop
  //

  for(size_t step = 0; step < steps; ++step){

    model.age++;

    //
    // Spawn Particles
    //

    spawn<<<block(n_samples, 512), 512>>>(particles, randStates, model.index);
    fill<<<block(model.discharge_track.elem(), 1024), 1024>>>(model.discharge_track, 0.0f);
    fill<<<block(model.suspended_track.elem(), 1024), 1024>>>(model.suspended_track, 0.0f);
    fill<<<block(model.momentum_track.elem(), 1024), 1024>>>(model.momentum_track, vec2(0.0f));
    cudaDeviceSynchronize();

    //
    // Erosion Loop
    //  1. Descend Particles (Accelerate, Move)
    //  2. Mass-Transfer
    //  3. Track

    descend<<<block(n_samples, 512), 512>>>(model, particles, param);

    //
    // Normalization and Filtering
    //

    // Normalize the Discharge by Sample Probability
    const float P = float(model.elem)/float(n_samples);
    normalize<<<block(model.elem, 1024), 1024>>>(model.discharge_track, 1.0f, P);

    // Filter the Result

    filter<<<block(model.elem, 1024), 1024>>>(model.discharge, model.discharge_track, param.lrate);
    filter<<<block(model.elem, 1024), 1024>>>(model.suspended, model.suspended_track, param.lrate);
    filter<<<block(model.elem, 1024), 1024>>>(model.momentum, model.momentum_track, param.lrate);
    cudaDeviceSynchronize();

    // apply the suspension difference...
    apply_height<<<block(model.elem, 1024), 1024>>>(model.height, model.suspended, param);

    // atomic add operations might still be coming in -
    // we have to be done before cascading or this fails...
    // we can't be computing the differences before they are determined...

    compute_cascade<<<block(model.elem, 1024), 1024>>>(model, model.discharge_track, param);
    apply_cascade<<<block(model.elem, 1024), 1024>>>(model, model.discharge_track, param);
    cudaDeviceSynchronize();

  }

  cudaFree(randStates);
}

} // end of namespace soil

#endif