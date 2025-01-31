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

__global__ void init_randstate(curandState* states, const size_t N, const size_t seed) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) return;
  curand_init(seed, n, 0, &states[n]);
}

__global__ void spawn(buffer_t<vec2> pos, curandState* randStates, flat_t<2> index){
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= pos.elem()) return;

  curandState* randState = &randStates[n];
  pos[n] = vec2{
    curand_uniform(randState)*float(index[0]),
    curand_uniform(randState)*float(index[1])
  };
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

__global__ void track(model_t model, soil::buffer_t<float> discharge_track, soil::buffer_t<vec2> momentum_track, particle_t particles){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= particles.elem) return;

  const ivec2 pos = particles.pos[ind];
  if(model.index.oob(pos)) return;

  // why would I have to scale this by lrate?

  const int find = model.index.flatten(pos);
  const float vol = particles.vol[ind];
  atomicAdd(&discharge_track[find], vol);   // Accumulate Current Volume into Tracking Buffer

  const vec2 m = vol * particles.spd[ind];
  atomicAdd(&momentum_track[find].x, m.x);
  atomicAdd(&momentum_track[find].y, m.y);

}

template<typename T>
__global__ void normalize(soil::buffer_t<T> out, const T B, const T P){
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= out.elem()) return;
  out[n] = B + P * out[n];
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

  // Retrieve Position, Check Bounds

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
  const vec2 average_speed = (model.momentum[find] + volume * speed) / (model.discharge[find] + volume);
  speed += param.momentumTransfer * average_speed; // note: assumes previous speed zero. needs fixing with dynamic time-step

  // Normalize Time-Step, Increment

  if(glm::length(speed) > 0.0){
    speed = sqrt(2.0f)*glm::normalize(speed);
  }

  // Compute Slope

  float h0 = model.height[find];
  float h1 = h0 - param.exitSlope; 
  if(!model.index.oob(pos + speed)){
    h1 = model.height[model.index.flatten(pos + speed)];
  }
  
  const float hdiff = (h0 - h1);

  // Tracking Part

  const float vol = particles.vol[ind];     // Water Volume
  const float sed = particles.sed[ind];     // Sediment Mass

  // Equilibrium Concentration
  // Note: Can't be Negative!
  const float discharge = log(1.0f + model.discharge[find]);
  const float c_eq = glm::max(hdiff, 0.0f) * (1.0f + discharge * param.entrainment);
  const float effD = param.depositionRate;

  float c_diff = (c_eq * vol - sed);
  if(effD * c_diff < -sed){
    c_diff = -sed / effD;
  }

  // Execute Mass-Transfer

  particles.sed[ind] += effD * c_diff;
  particles.vol[ind] *= (1.0f - param.evapRate);
  atomicAdd(&model.height[find], -effD * c_diff);

  particles.spd[ind] = speed;
  particles.pos[ind] += speed;

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

  soil::buffer_t<float> discharge_track(model.discharge.elem(), soil::host_t::GPU);
  soil::buffer_t<vec2> momentum_track(model.momentum.elem(), soil::host_t::GPU);

  //! \todo remove this allocation, as well as the randstate allocation
  particle_t particles{n_samples};

  //
  // Initialize Rand-State Buffer
  //

  curandState* randStates;
  cudaMalloc((void**)&randStates, n_samples * sizeof(curandState));
  init_randstate<<<block(n_samples, 512), 512>>>(randStates, n_samples, 0);

  cudaDeviceSynchronize();

  //
  // Execute Erosion Loop
  //

  for(size_t step = 0; step < steps; ++step){

    //
    // Spawn Particles
    //

    spawn<<<block(n_samples, 512), 512>>>(particles.pos, randStates, model.index);
    fill<<<block(n_samples, 512), 512>>>(particles.spd, vec2(0.0f));
    fill<<<block(n_samples, 512), 512>>>(particles.vol, 1.0f);
    fill<<<block(n_samples, 512), 512>>>(particles.sed, 0.0f);

    fill<<<block(discharge_track.elem(), 1024), 1024>>>(discharge_track, 0.0f);
    fill<<<block(momentum_track.elem(), 1024), 1024>>>(momentum_track, vec2(0.0f));
    cudaDeviceSynchronize();

    //
    // Erosion Loop
    //  1. Descend Particles (Accelerate, Move)
    //  2. Mass-Transfer
    //  3. Track

    for(size_t age = 0; age < param.maxage; ++age){

      descend<<<block(n_samples, 512), 512>>>(model, particles, param);
      track<<<block(n_samples, 512), 512>>>(model, discharge_track, momentum_track, particles);

    }

//    // We have to add the excess sediment...
    // dump<<<block(n_particles, 512), 512>>>(model, particles, param);

    //
    // Normalization and Filtering
    //

    // Normalize the Discharge by Sample Probability
    const float P = float(model.elem)/float(n_samples);
    normalize<<<block(model.elem, 1024), 1024>>>(discharge_track, 1.0f, P);

    // Filter the Result

    filter<<<block(model.elem, 1024), 1024>>>(model.discharge, discharge_track, param.lrate);
    filter<<<block(model.elem, 1024), 1024>>>(model.momentum, momentum_track, param.lrate);
    cudaDeviceSynchronize();

    // atomic add operations might still be coming in -
    // we have to be done before cascading or this fails...
    // we can't be computing the differences before they are determined...

    compute_cascade<<<block(model.elem, 1024), 1024>>>(model, discharge_track, param);
    apply_cascade<<<block(model.elem, 1024), 1024>>>(model, discharge_track, param);
    cudaDeviceSynchronize();

  }

}

} // end of namespace soil

#endif