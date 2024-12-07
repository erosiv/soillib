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

__global__ void descend(const model_t model, particle_t particles){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= particles.elem) return;

  // Retrieve Position, Check Bounds

  const ivec2 pos = particles.pos[ind];
  if(model.index.oob(pos))
    return;

  const int find = model.index.flatten(pos);

  // Skip Depleted Particles

  const float volume = particles.vol[ind];
  const float minVol = 0.001;
  if (volume < minVol) {
    return;
  }

  // Compute Speed Update

  vec2 speed = particles.spd[ind];

  // Gravity Contribution
  // Compute Normal Vector

  sample_t<float> px[5], py[5];
  gather<float, soil::flat_t<2>>(model.height, model.index, ivec2(pos), px, py);
  const vec2 grad = gradient_detailed<float>(px, py);
  const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));

  const float gravity = 2.0f;
  speed += gravity * vec2(normal.x, normal.y) / volume;

  // Momentum Transfer

  const vec2 fspeed = model.momentum[find];
  const float discharge = erf(0.4f * model.discharge[find]);
  const float momentumTransfer = 0.1f;
  if (glm::length(fspeed) > 0 && glm::length(speed) > 0)
    speed += momentumTransfer * glm::dot(glm::normalize(fspeed), glm::normalize(speed)) / (volume + discharge) * fspeed;
  
  // Normalize Time-Step, Increment
  
  if(glm::length(speed) > 0.0f){
    speed = sqrtf(2.0f) * glm::normalize(speed);
  }

  //const float hdiff = hdiff_buf[ind];
  //float h0 = model.height[find];
  //float h1 = 0.99f*h0;
  //if(!model.index.oob(pos + speed)){
  //  h1 = model.height[model.index.flatten(pos + speed)];
  //}
  //particles.slope[ind] = (h0 - h1);
  
  particles.spd[ind] = speed;
  particles.pos[ind] += speed;
}

__global__ void track(model_t model, particle_t particles){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= particles.elem) return;

  const ivec2 pos = particles.pos[ind];
  if(model.index.oob(pos)) return;

  const int find = model.index.flatten(pos);
  const float vol = particles.vol[ind];
  atomicAdd(&model.discharge[find], vol);

  const vec2 m = vol * particles.spd[ind];
  atomicAdd(&model.momentum[find].x, m.x);
  atomicAdd(&model.momentum[find].y, m.y);

}

__global__ void transfer(model_t model, particle_t particles){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= particles.elem) return;

  const float evapRate = 0.001f;
  const float depositionRate = 0.05f;

  const vec2 speed = particles.spd[ind];  // Current Speed
  const vec2 pos1 = particles.pos[ind];   // Current Position
  const vec2 pos0 = pos1 - speed;         // Old Position

  if(model.index.oob(ivec2(pos0))) return;

  // Compute Equilibrium Mass-Transfer

  //const float hdiff = hdiff_buf[ind];
  float h0 = model.height[model.index.flatten(pos0)];
  float h1 = 0.99f*h0;
  if(!model.index.oob(pos1)){
    h1 = model.height[model.index.flatten(pos1)];
  }

  float hdiff = (h0 - h1);

  const float vol = particles.vol[ind]; // Water Volume
  float sed = particles.sed[ind];       // Sediment Mass

  // Equilibrium Concentration
  // Note: Can't be Negative!
  const float entrainment = 0.0f;
  const float discharge = erf(0.4f *model.discharge[model.index.flatten(pos0)]);
  
  const float c_eq = glm::max(hdiff, 0.0f) * (1.0f + discharge * entrainment);
  const float effD = depositionRate;

  float c_diff = (c_eq * vol - sed);
  if(isnan(c_diff) || isinf(c_diff)){
    c_diff = 0.0f;
  }

  if(isnan(sed) || sed < 0.0f || isinf(sed)){
    sed = 0.0f; 
  }

  // can only give as much mass as we have...
  if(effD * c_diff < -sed){
    c_diff = -sed / effD;
  }

  // Execute Mass-Transfer
  const int find = model.index.flatten(ivec2(pos0));

  //!\todo figure out why find zero gives so many problems...
  // why would this every be a problem? I don't get it...
//  if(find != 0){

    particles.sed[ind] += effD * c_diff;
    particles.vol[ind] *= (1.0f - evapRate);

    atomicAdd(&model.height[find], -effD * c_diff);

// }
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

__global__ void clamp(soil::buffer_t<float> height){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= height.elem()) return;

  if(height[ind] > 256) height[ind] = 256;
  if(height[ind] < -256) height[ind] = -256;

}

void gpu_erode(model_t& model, const size_t steps, const size_t maxage){

  std::cout<<"Launched GPU Erode Kernel"<<std::endl;

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

  std::cout<<"Setting Up Particle Buffers..."<<std::endl;

  soil::buffer_t<float> discharge_track(model.discharge.elem(), soil::host_t::GPU);
  soil::buffer_t<vec2> momentum_track(model.momentum.elem(), soil::host_t::GPU);

  const size_t n_particles = 1024;
  particle_t particles{n_particles};

  //
  // Initialize Rand-State Buffer
  //

  std::cout<<"Initializing Random State..."<<std::endl;

  curandState* randStates;
  cudaMalloc((void**)&randStates, n_particles * sizeof(curandState));
  init_randstate<<<block(n_particles, 512), n_particles>>>(randStates, n_particles, 0);

  cudaDeviceSynchronize();

  //
  // Execute Erosion Loop
  //

  std::cout<<"Eroding..."<<std::endl;

  for(size_t step = 0; step < steps; ++step){

    //
    // Spawn Particles
    //

    spawn<<<block(n_particles, 512), n_particles>>>(particles.pos, randStates, model.index);
    fill<<<block(n_particles, 512), n_particles>>>(particles.spd, vec2(0.0f));
    fill<<<block(n_particles, 512), n_particles>>>(particles.vol, 1.0f);
    fill<<<block(n_particles, 512), n_particles>>>(particles.sed, 0.0f);
//    fill<<<block(n_particles, 512), n_particles>>>(particles.slope, 0.0f);

    fill<<<block(discharge_track.elem(), 1024), 1024>>>(discharge_track, 0.0f);
    fill<<<block(momentum_track.elem(), 1024), 1024>>>(momentum_track, vec2(0.0f));
    cudaDeviceSynchronize();

    //
    // Erosion Loop
    //  1. Descend Particles (Accelerate, Move)
    //  2. Mass-Transfer
    //  3. Track

    for(size_t age = 0; age < maxage; ++age){

      descend<<<block(n_particles, 512), 512>>>(model, particles);
      cudaDeviceSynchronize();

      track<<<block(n_particles, 512), 512>>>(model, particles);
      cudaDeviceSynchronize();

      transfer<<<block(n_particles, 512), 512>>>(model, particles);
      cudaDeviceSynchronize();

    }

    filter<<<block(model.elem, 1024), 1024>>>(model.discharge, discharge_track, 0.01f);
    filter<<<block(model.elem, 1024), 1024>>>(model.momentum, momentum_track, 0.01f);
    cudaDeviceSynchronize();
  
  }

  // necessary solution to temporarily fix an indexing problem
  // which is introducing unrealistically large values into the
  // height buffer - who knows why.

  clamp<<<block(model.elem, 1024), 1024>>>(model.height);

}

} // end of namespace soil

#endif