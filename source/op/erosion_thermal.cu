#ifndef SOILLIB_OP_EROSION_THERMAL_CU
#define SOILLIB_OP_EROSION_THERMAL_CU
#define HAS_CUDA

#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <soillib/op/erosion.hpp>
#include <soillib/op/gather.hpp>

//! Thermal Erosion Algorithms
//!
//! These device functions implement various methods to compute
//! the time-differential of a height-map from thermal erosion.
//! 
//! \todo Research alternative thermal erosion implementations

namespace soil {

//! Slope Limiting Mass Wasting
__device__ void slope_limit(const buffer_t<float>& height, buffer_t<float>& height_diff, const flat_t<2>& index, const ivec2 ipos, const param_t param, const vec3 scale) {

  if(index.oob(ipos))
    return;

  // Get Non-Out-of-Bounds Neighbors

  const ivec2 n[] = {
    ivec2(-1, -1),
    ivec2(-1, 0),
    ivec2(-1, 1),
    ivec2(0, -1),
    ivec2(0, 1),
    ivec2(1, -1),
    ivec2(1, 0),
    ivec2(1, 1)
  };

  const float w[] = {
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  };

  struct Point {
    float h;
    float d;
  } sn[8];

  int num = 0;

  for(auto &nn : n){

    ivec2 npos = ipos + nn;

    if(index.oob(npos))
      continue;

    const size_t i = index.flatten(npos);
    sn[num] = {height[i]*scale.y, length(vec2(scale.x, scale.z)*vec2(nn))};
    ++num;
  }

  const size_t i = index.flatten(ipos);
  const float h = height[i]*scale.y;
  float transfer_tot = 0.0f; 
  
  for(int i = 0; i < num; ++i){

    // Full Height-Different Between Positions!
    float diff = h - sn[i].h;
    if (diff == 0) // No Height Difference
      continue;

    // The Amount of Excess Difference!
    // Note: Maxslope is a slope value.
    //  By scaling it by the correct distance value in world-space,
    //  we get a height value in world-space. This is the value which
    //  we subtract to get the excess.
    float excess = 0.0f;
    excess = abs(diff) - sn[i].d * w[i]* param.maxdiff;
    if (excess <= 0) // No Excess
      continue;

    excess = (diff > 0) ? -excess : excess;

    // Actual Amount Transferred
    float transfer = param.settling * excess / 2.0f;
    transfer_tot += transfer;
  }

  height_diff[i] = transfer_tot / (float) num;

}

//! Laplacian Based Mass Wasting Function
//!
__device__ void thermal_laplacian(const buffer_t<float>& height, buffer_t<float>& height_diff, const flat_t<2>& index, const ivec2 ipos, const param_t param) {

  if(index.oob(ipos))
    return;

  const size_t i = index.flatten(ipos + ivec2( 0, 0));
  float ct = height[i];
  float nx = ct;
  float ny = ct;
  float px = ct;
  float py = ct;
  if(!index.oob(ipos + ivec2( 1, 0))) px = height[index.flatten(ipos + ivec2( 1, 0))];
  if(!index.oob(ipos + ivec2(-1, 0))) nx = height[index.flatten(ipos + ivec2(-1, 0))];
  if(!index.oob(ipos + ivec2( 0, 1))) py = height[index.flatten(ipos + ivec2( 0, 1))];
  if(!index.oob(ipos + ivec2( 0,-1))) ny = height[index.flatten(ipos + ivec2( 0,-1))];

  float lx = px + nx - 2.0f * ct;
  float ly = py + ny - 2.0f * ct;
  float L = lx + ly;
  if(glm::abs(L) < param.maxdiff) L = 0.0f;
  height_diff[i] = param.settling * L;

}

//
// Thermal Erosion Kernels
//

__global__ void compute_cascade(model_t model, buffer_t<float> transfer, const param_t param) {

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= model.elem) return;
  const ivec2 ipos = model.index.unflatten(ind);

  slope_limit(model.height, transfer, model.index, ipos, param, model.scale);
  //thermal_laplacian(model.height, transfer, model.index, ipos, param);

}

__global__ void apply_cascade(model_t model, buffer_t<float> transfer_b, const param_t param){
  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= model.elem) return;

  // Only cascade where agitation exists?

  const float transfer = transfer_b[ind] / model.scale.y;
  const float discharge = log(1.0f + model.discharge[ind])/6.0f;
  model.height[ind] += discharge * transfer;
  //model.height[ind] += transfer;
}

//
// Thermal Erosion Particle
//

__device__ float local_laplacian(const ivec2 pos, const buffer_t<float>& height, const flat_t<2>& index){

  const size_t i = index.flatten(pos + ivec2( 0, 0));
  float ct = height[i];
  float nx = ct;
  float ny = ct;
  float px = ct;
  float py = ct;
  if(!index.oob(pos + ivec2( 1, 0))) px = height[index.flatten(pos + ivec2( 1, 0))];
  if(!index.oob(pos + ivec2(-1, 0))) nx = height[index.flatten(pos + ivec2(-1, 0))];
  if(!index.oob(pos + ivec2( 0, 1))) py = height[index.flatten(pos + ivec2( 0, 1))];
  if(!index.oob(pos + ivec2( 0,-1))) ny = height[index.flatten(pos + ivec2( 0,-1))];

  float lx = px + nx - 2.0f * ct;
  float ly = py + ny - 2.0f * ct;
  return lx + ly;

}

// Note: I should try to implement this using a stability function instead
// Stability Function Based Erosion:
//  A particle is spawnd at a specific location and computes the local
//  stable mass height, based on the surrounding mass. Then, it has a
//  probability to knock that mass loose. The mass descends by gravity
//  and momentum. As it descends, it computes the next mass stability
//  function and so on, accumulating and depositing a stable mass.

__global__ void thermal_particle(model_t model, const size_t N, const param_t param){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= model.elem) return;

  // Parameters

  const vec3 scale = model.scale;
  const float g = param.gravity;
  const float P = float(model.elem)/float(N); // Sample Probability

  // Spawn Particle at Random Position

  curandState* randState = &model.rand[ind];
  vec2 pos = vec2{
    curand_uniform(randState)*float(model.index[0]),
    curand_uniform(randState)*float(model.index[1])
  };

  // Compute Stable Slope Height at Position
  // Question: Can this be implemented as a
  //  non-random sampling but rather a kernel,
  //  which instead uses a thermal erosion probability?

  // Note: Using a simple 8x8 Lowest leads to weird looking
  //  results. The reason is that if only one is low, but
  //  all others are high, the interpolated structure looks
  //  like a pit has been created.

  const ivec2 n[] = {
//    ivec2(-1, -1),
    ivec2(-1,  0),
//    ivec2(-1,  1),
    ivec2( 0, -1),
    ivec2( 0,  1),
//    ivec2( 1, -1),
    ivec2( 1,  0),
//    ivec2( 1,  1)
  };

  struct Point {
    float h;
    float d;
  } sn[4];
  
  const size_t i = model.index.flatten(pos);
  
  float h = model.height[i]*scale.y;
  float mass = 0.0f;  // Currently Transported Mass

  // Iterate over a Number of Steps

  for(size_t age = 0; age < 32; ++age){

    // Note: This slope stability function can be replaced
    //  with something more complex in general!
    // The stable height becomes the lowst stable height

    float stable = h; // Stable Height at Position
    int lowest = 0;

    int num = 0;
    for(auto &nn : n){
      ivec2 npos = ivec2(pos) + nn;
      if(model.index.oob(npos))
        continue;

      const size_t i = model.index.flatten(npos);
      sn[num] = {model.height[i]*scale.y, length(vec2(scale.x, scale.z)*vec2(nn))};
      ++num;

    }

    for(int i = 0; i < num; ++i){
      float next_stable = sn[i].h + sn[i].d * param.maxdiff;
      if(next_stable < stable){
        stable = next_stable;
        lowest = i;
      }
    }

    // We compute the difference to the stable amount
    float excess = h - stable;

    // if excess is less than zero,
    //  that means that we can add mass if we have any!
    if(excess < 0.0f){
      float transfer = glm::min(-excess, mass);
      int find = model.index.flatten(pos);
      atomicAdd(&model.height[find], param.settling * transfer);
      mass -= param.settling * transfer;
    }

    // if it is equal zero, we are exactly stable

    else if(excess > 0.0f){

      float transfer = excess;
      int find = model.index.flatten(pos);
      atomicAdd(&model.height[find], -param.settling * transfer);
      mass += param.settling * transfer;
    }

    if(mass == 0.0f)
      break;

    // finally, we must move to the next position
    // how do we determine the next position?

    // Next Position:
    lerp5_t<float> lerp;
    lerp.gather(model.height, model.index, ivec2(pos));
    const vec2 grad = lerp.grad(model.scale);

    const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
    vec2 speed = g * vec2(normal.x, normal.y);
    if(glm::length(speed) < 0.001f)
      return;
    else pos += glm::normalize(speed);

    if(model.index.oob(pos)){
      return;
    }

    const size_t i = model.index.flatten(pos);
    h = model.height[i]*scale.y;

  }

  if(!model.index.oob(pos)){
    const size_t i = model.index.flatten(pos);
    atomicAdd(&model.height[i], mass);
  }

}


} // end of namespace soil

#endif