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

namespace soil {

namespace {

int block(const int elem, const int thread){
  return (elem + thread - 1)/thread;
}

}

__global__ void init_randstate(curandState* states, const size_t N, const size_t seed) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= N) return;
  curand_init(seed, index, 0, &states[index]);
}

__global__ void spawn(buffer_t<vec2> pos_buf, curandState* randStates, flat_t<2> index){

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= pos_buf.size()) return;

  curandState* randState = &randStates[n];
  vec2 pos {
    curand_uniform(randState)*float(index[0]),
    curand_uniform(randState)*float(index[1])
  };

  pos_buf[n] = pos;

}

template<typename T>
__global__ void fill(soil::buffer_t<T> buf, const T val){
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < buf.size())
    buf[index] = val;
}

__global__ void descend(const soil::buffer_t<float> height, const soil::flat_t<2> index, soil::buffer_t<vec2> pos, soil::buffer_t<vec2> speed, soil::buffer_t<float> vol_b, soil::buffer_t<float> sed_b){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= pos.elem()) return;

  if(index.oob(pos[ind])){
//    speed[ind] = vec2(0);
//    vol_b[ind] = 0.0f;
//    sed_b[ind] = 0.0f;
    return;
  }

  if(oob(pos[ind], index)){
//    speed[ind] = vec2(0);
//    vol_b[ind] = 0.0f;
//    sed_b[ind] = 0.0f;
    return;
  }

  const lerp_t<float> lerp = gather(height, index, pos[ind]);
  const vec2 grad = lerp.grad();
  const vec3 n = glm::normalize(vec3(-grad.x, -grad.y, 1.0));

//  vec2 s = speed[ind];
//  s += 2.0f * vec2(n);
//  if(glm::length(s) > 0.0f){
//    s = sqrtf(2.0f) * glm::normalize(s);
//  }

  vec2 s = sqrtf(2.0f) * glm::normalize(vec2(n.x, n.y));
  speed[ind] = s;
  pos[ind] += s;

}

__global__ void _discharge(soil::buffer_t<float> discharge, const soil::flat_t<2> index, soil::buffer_t<vec2> pos, soil::buffer_t<float> vol_b){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= pos.elem()) return;

  if(oob(pos[ind], index)) 
    return;

  const int find = index.flatten(pos[ind]);
  const float vol = vol_b[ind];
  atomicAdd(&discharge[find], vol);

}

__global__ void transfer(soil::buffer_t<float> height, const soil::flat_t<2> index, soil::buffer_t<vec2> pos_b, soil::buffer_t<vec2> speed_b, soil::buffer_t<float> vol_b, soil::buffer_t<float> sed_b){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= pos_b.elem()) return;

  const float evapRate = 0.001f;
  const float depositionRate = 0.05f;

  const vec2 speed = speed_b[ind];  // Current Speed
  const vec2 pos1 = pos_b[ind];     // Current Position
  const vec2 pos0 = pos1 - speed;   // Old Position

  if(index.oob(pos0)) return;

  // Sample Height Values (Old Position, New Position)
  
  float h0 = height[index.flatten(pos0)];
  float h1 = 0.99f*h0;
  if(!index.oob(pos1)){
    h1 = height[index.flatten(pos1)];
  }

  if(isnan(h0) || isnan(h1)){
    return;
  }

  // Compute Equilibrium Mass-Transfer

  const float vol = vol_b[ind]; // Water Volume
  const float sed = sed_b[ind]; // Sediment Mass

  // Equilibrium Concentration
  // Note: Can't be Negative!
  const float c_eq = glm::max(h0 - h1, 0.0f);
  const float effD = depositionRate;

  float c_diff = (c_eq * vol - sed);
  if(isnan(c_diff)){
    c_diff = 0.0f;
  }

  // can only give as much mass as we have...
  if(effD * c_diff < -sed){
    c_diff = -sed / effD;
  }

  // Execute Mass-Transfer
  const int find = index.flatten(ivec2(pos0));

  //!\todo figure out why find zero gives so many problems...
  // why would this every be a problem? I don't get it...
  if(find != 0){

    sed_b[ind] += effD * c_diff;
    vol_b[ind] *= (1.0f - evapRate);

    atomicAdd(&height[find], -effD * c_diff);

  }
}

__global__ void clamp(soil::buffer_t<float> height){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= height.elem()) return;

  if(height[ind] > 256) height[ind] = 256;
  if(height[ind] < -256) height[ind] = -256;

}

void gpu_erode(soil::buffer &buffer, soil::buffer& discharge, const soil::index &index, const size_t steps, const size_t maxage){

  std::cout<<"Launched GPU Erode Kernel"<<std::endl;

  if(buffer.host() != soil::host_t::GPU){
    throw soil::error::mismatch_host(soil::host_t::GPU, buffer.host());
  }

  //
  // Type-Cast Buffers
  //

  auto buffer_t = buffer.as<float>();
  auto index_t = index.as<flat_t<2>>();

  auto discharge_t = discharge.as<float>();

  //
  // Particle Buffers
  //

  std::cout<<"Setting Up Particle Buffers..."<<std::endl;

  const size_t n_particles = 1024;

  soil::buffer_t<vec2> pos_buf(n_particles, soil::host_t::GPU);
  soil::buffer_t<vec2> spd_buf(n_particles, soil::host_t::GPU);

  soil::buffer_t<float> vol_buf(n_particles, soil::host_t::GPU);
  soil::buffer_t<float> sed_buf(n_particles, soil::host_t::GPU);

  //
  // Initialize Rand-State Buffer
  //

  std::cout<<"Initializing Random State..."<<std::endl;

  curandState* randStates;
  cudaMalloc((void**)&randStates, n_particles * sizeof(curandState));
  init_randstate<<<block(n_particles, 512), n_particles>>>(randStates, n_particles, 0);

  //
  // Execute Erosion Loop
  //

  std::cout<<"Eroding..."<<std::endl;

  for(size_t step = 0; step < steps; ++step){

    //
    // Spawn Particles
    //

    spawn<<<block(n_particles, 512), n_particles>>>(pos_buf, randStates, index.as<flat_t<2>>());
    fill<<<block(n_particles, 512), n_particles>>>(spd_buf, vec2(0.0f));
    fill<<<block(n_particles, 512), n_particles>>>(vol_buf, 1.0f);
    fill<<<block(n_particles, 512), n_particles>>>(sed_buf, 0.0f);
    fill<<<block(n_particles, 512), n_particles>>>(discharge_t, 0.0f);

    //
    // Erosion Loop
    //  1. Descend Particles (Accelerate, Move)
    //  2. Mass-Transfer
    //  3. Track

    for(size_t age = 0; age < maxage; ++age){

      descend<<<block(n_particles, 512), 512>>>(buffer_t, index_t, pos_buf, spd_buf, vol_buf, sed_buf);
      // _discharge<<<block(n_particles, 512), 512>>>(discharge_t, index_t, pos_buf, sed_buf);
      transfer<<<block(n_particles, 512), 512>>>(buffer_t, index_t, pos_buf, spd_buf, vol_buf, sed_buf);

    }

  }

  // necessary solution to temporarily fix an indexing problem
  // which is introducing unrealistically large values into the
  // height buffer - who knows why.

  clamp<<<block(buffer_t.elem(), 1024), 1024>>>(buffer_t);

  // Loop for Number of Steps per Particle:
  // 1. Accelerate Particles
  // 2. 

  //
  // Note: In principle we can use an age buffer,
  //  or a termination check, to see if the particle
  //  gets re-spawned immediately and we just keep looping...

}

} // end of namespace soil

#endif