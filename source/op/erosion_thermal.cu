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

  /*
  const float w[] = {
    0.4f,
    0.4f,
    0.4f,
    2.0f,
    2.0f,
    4.0f,
    4.0f,
    4.0f
  };
  */

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

  const float P = float(model.elem)/float(N); // Sample Probability
  curandState* randState = &model.rand[ind];
  vec2 pos = vec2{
    curand_uniform(randState)*float(model.index[0]),
    curand_uniform(randState)*float(model.index[1])
  };

  /*
  auto index = model.index;
  auto height = model.height;

  const size_t i = index.flatten(pos);
  float ct = height[i];
  float nx = ct;
  float ny = ct;
  float px = ct;
  float py = ct;
  if(!index.oob(pos + vec2( 1, 0))) px = height[index.flatten(pos + vec2( 1, 0))];
  if(!index.oob(pos + vec2(-1, 0))) nx = height[index.flatten(pos + vec2(-1, 0))];
  if(!index.oob(pos + vec2( 0, 1))) py = height[index.flatten(pos + vec2( 0, 1))];
  if(!index.oob(pos + vec2( 0,-1))) ny = height[index.flatten(pos + vec2( 0,-1))];

  float lx = px + nx - 2.0f * ct;
  float ly = py + ny - 2.0f * ct;
  float L = lx + ly;
  */

  // 
  float L = local_laplacian(pos, model.height, model.index);
  // Note: If Laplacian (Curvature) is larger than zero, this means that on average,
  //  the central value is lower than the surrounding values.
  // we don't change the mass of locations in pits
  if(L > 0.0f) return;
  // Threshold Laplacian
  // i.e. value has to be "negative enough"
  if(glm::abs(L) < param.maxdiff) 
    return;

  const float g = param.gravity;

  int find = model.index.flatten(pos);

  // Remove the Mass from the Height-Map


  // Descend the Mass, set it down somewhere else
  lerp5_t<float> lerp;
  lerp.gather(model.height, model.index, ivec2(pos));
  const vec2 grad = lerp.grad(model.scale);

  // We need slope ...
  // we need stable mass and we need unstable mass
  // unstable mass is what is knocked loose
  // and we always deposit the excess stable mass

  const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
  vec2 speed = g * vec2(normal.x, normal.y);
  if(glm::length(speed) < 0.01f){
    return;
  }

  // we don't do this where it isn't steep!
//  if(normal.z > 0.75f){
//    return;
//  }

  float mass = -param.settling * L;
  atomicAdd(&model.height[find], -mass);

  for(size_t age = 0; age < 32; ++age){

    if(model.index.oob(pos)) return;
//    if(glm::length(speed) < 1E-2) return;

    lerp5_t<float> lerp;
    lerp.gather(model.height, model.index, ivec2(pos));
    const vec2 grad = lerp.grad(model.scale);

    const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
    vec2 nspeed = g * vec2(normal.x, normal.y);

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

    if(!model.index.oob(npos)){
      find = model.index.flatten(npos);
      atomicAdd(&model.height[find], mass / 32.0f);
    }

    pos = npos;
    speed = nspeed;

  }

}


} // end of namespace soil

#endif