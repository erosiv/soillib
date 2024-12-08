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

//
// Cascading Kernel
//
// Effectively we have to compute the height-difference between every cell
// and its non-out-of-bounds neighbors, then we have to transfer the sediment.
// How do we do this without race conditions?
// I suppose that we need an additional buffer to determine the updated sediment amounts...
// so that we can ping-pong back and forth...

// for now, we will implement this as a device function locally and perhaps switch to
// a singular kernel later.

// __global__ void cascade(model_t model){
// }

__device__ void cascade(model_t& model, const glm::ivec2 ipos) {

  if(model.index.oob(ipos))
    return;

  // Get Non-Out-of-Bounds Neighbors

  static const glm::ivec2 n[] = {
      glm::ivec2(-1, -1),
      glm::ivec2(-1, 0),
      glm::ivec2(-1, 1),
      glm::ivec2(0, -1),
      glm::ivec2(0, 1),
      glm::ivec2(1, -1),
      glm::ivec2(1, 0),
      glm::ivec2(1, 1)
  };

  struct Point {
    glm::ivec2 pos;
    float h;
    float d;
  } sn[8];

  int num = 0;

  for(auto &nn : n){

    glm::ivec2 npos = ipos + nn;

    if (model.index.oob(npos))
      continue;

    const size_t index = model.index.flatten(npos);
    const float height = model.height[index];
    sn[num++] = {npos, height, length(glm::vec2(nn))};
  }

  const size_t index = model.index.flatten(ipos);
  const float height = model.height[index];
  float h_ave = height;
  for (int i = 0; i < num; ++i)
    h_ave += sn[i].h;
  h_ave /= (float)(num + 1);

  for (int i = 0; i < num; ++i) {

    // Full Height-Different Between Positions!
    float diff = h_ave - sn[i].h;
    if (diff == 0) // No Height Difference
      continue;

    const glm::ivec2 &tpos = (diff > 0) ? ipos : sn[i].pos;
    const glm::ivec2 &bpos = (diff > 0) ? sn[i].pos : ipos;

    const size_t tindex = model.index.flatten(tpos);
    const size_t bindex = model.index.flatten(bpos);

    const float maxdiff = 0.8f;
    const float settling = 1.0f;

    // The Amount of Excess Difference!
    float excess = 0.0f;
    excess = abs(diff) - sn[i].d * maxdiff;
    if (excess <= 0) // No Excess
      continue;

    // Actual Amount Transferred
    float transfer = settling * excess / 2.0f;

    atomicAdd(&model.height[tindex], -transfer);
    atomicAdd(&model.height[bindex], transfer);
  }
}

//
// Erosion Kernels
//

__global__ void descend(const model_t model, particle_t particles){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= particles.elem) return;

  // Retrieve Position, Check Bounds

  const vec2 pos = particles.pos[ind];
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
  const float momentumTransfer = 1.0f;
  if (glm::length(fspeed) > 0 && glm::length(speed) > 0)
    speed += momentumTransfer * glm::dot(glm::normalize(fspeed), glm::normalize(speed)) / (volume + discharge) * fspeed;
  
  // Normalize Time-Step, Increment
  
  if(glm::length(speed) > 0.0f){
    speed = sqrtf(2.0f) * glm::normalize(speed);
  }

  // Compute Slope

  particles.spd[ind] = speed;

  float h0 = model.height[find];
  float h1 = 0.99f*h0;
  if(!model.index.oob(pos + speed)){
    h1 = model.height[model.index.flatten(pos + speed)];
  }
  particles.slope[ind] = (h0 - h1);
}

__global__ void transfer(model_t model, particle_t particles){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= particles.elem) return;

  const vec2 pos = particles.pos[ind];    // Current Position
  if(model.index.oob(pos))
    return;
  
  const int find = model.index.flatten(pos);

  // Compute Equilibrium Mass-Transfer

  const vec2 speed = particles.spd[ind];    // Current Speed
  const float hdiff = particles.slope[ind]; // Local Slope
  const float vol = particles.vol[ind];     // Water Volume
  const float sed = particles.sed[ind];     // Sediment Mass

  const float evapRate = 0.001f;
  const float depositionRate = 0.05f;
  const float entrainment = 4.0f;

  // Equilibrium Concentration
  // Note: Can't be Negative!
  const float discharge = erf(0.4f *model.discharge[find]);
  
  const float c_eq = glm::max(hdiff, 0.0f) * (1.0f + discharge * entrainment);
  const float effD = depositionRate;

  float c_diff = (c_eq * vol - sed);
  if(effD * c_diff < -sed){
    c_diff = -sed / effD;
  }

  // Execute Mass-Transfer

  particles.sed[ind] += effD * c_diff;
  particles.vol[ind] *= (1.0f - evapRate);
  atomicAdd(&model.height[find], -effD * c_diff);

  cascade(model, pos); // let's see if this works...

  particles.pos[ind] += speed;
}

void gpu_erode(model_t& model, const size_t steps, const size_t maxage){

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
  const size_t n_particles = 512;
  particle_t particles{n_particles};

  //
  // Initialize Rand-State Buffer
  //

  curandState* randStates;
  cudaMalloc((void**)&randStates, n_particles * sizeof(curandState));
  init_randstate<<<block(n_particles, 512), n_particles>>>(randStates, n_particles, 0);

  cudaDeviceSynchronize();

  //
  // Execute Erosion Loop
  //

  for(size_t step = 0; step < steps; ++step){

    //
    // Spawn Particles
    //

    spawn<<<block(n_particles, 512), n_particles>>>(particles.pos, randStates, model.index);
    fill<<<block(n_particles, 512), n_particles>>>(particles.spd, vec2(0.0f));
    fill<<<block(n_particles, 512), n_particles>>>(particles.vol, 1.0f);
    fill<<<block(n_particles, 512), n_particles>>>(particles.sed, 0.0f);
    fill<<<block(n_particles, 512), n_particles>>>(particles.slope, 0.0f);

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
      transfer<<<block(n_particles, 512), 512>>>(model, particles);
      track<<<block(n_particles, 512), 512>>>(model, particles);

    }

    filter<<<block(model.elem, 1024), 1024>>>(model.discharge, discharge_track, 0.01f);
    filter<<<block(model.elem, 1024), 1024>>>(model.momentum, momentum_track, 0.01f);
    cudaDeviceSynchronize();
  
  }

}

} // end of namespace soil

#endif