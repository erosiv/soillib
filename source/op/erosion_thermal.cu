#ifndef SOILLIB_OP_EROSION_THERMAL_CU
#define SOILLIB_OP_EROSION_THERMAL_CU
#define HAS_CUDA

#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <soillib/op/erosion.hpp>
#include <soillib/op/gather.hpp>

namespace soil {

//! Thermal Erosion / Debris Flow Algorithm
//!
//! Bank-Stability Function Based Debris Flow Method:
//!
//!   The total debris flow is computed along characteristics,
//!   where the mass contribution is given by the excess mass.
//!   The bank-stability function determines what the theoretical stable
//!   bank height is, and the excess is the difference to this value.
//!   Note that this is effectively an equilibrium model.
//!
//!   The bank stability function is computed along the direction
//!   of gravity acting on the surface normal. The mass is then moved
//!   along this direction.
//!
//!   Solved using the path-integral method, the scale of the equilibrium
//!   constant also corresponds to the rate of thermal cracking events.

//! Steepest Direction Computed by Surface Normal
//!
//! Note: Normally the normal vector would be computed instead of just the
//!   gradient, and scaled by the gravitational constant to yield the correct
//!   acceleration. Since we are using a dynamic time-step, this is normalized
//!   away. If we compute the acceleration and limiting slope based on inter-
//!   particle frictions though (e.g. for multi-material interfaces), then the
//!   term does NOT becomes normalized away and becomes relevant again.
//! 
__device__ vec2 steepest(model_t& model, const ivec2 pos) {

  lerp5_t<float> lerp;
  lerp.gather(model.height, model.index, pos);
  vec2 dir = -1.0f * lerp.grad(model.scale);

  if(glm::length(dir) < 0.00001f)
    return vec2(0.0f);
  else return sqrt(2.0f) * glm::normalize(dir);

}

__device__ float stable_height(model_t& model, const ivec2 pos, const param_t param, const float height, const vec2 speed) {

  // Boundary Condition:
  const vec2 npos = vec2(pos) + speed;
  if(model.index.oob(npos) || glm::length(speed) == 0.0f){
    return height;
  }

  float dist = glm::length(speed * vec2(model.scale.x, model.scale.z));
  float h_next = model.height[model.index.flatten(npos)]*model.scale.y;
  float stable = h_next + param.maxdiff*dist;
  return stable;

}

/*
//! Debris Flow Kernel Implementation
//!
__global__ void debris_flow(model_t model, const size_t N, const param_t param){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= model.elem) return;

  // Parameters

  const vec3 scale = model.scale; // Local Physical Scale
  const float g = param.gravity;  // Gravitational Constant

  // Spawn Particle at Random Position

  curandState* randState = &model.rand[ind];  // Local Randstate
  const float P = float(model.elem)/float(N); // Sampling Probability
  vec2 pos = vec2 {
    curand_uniform(randState)*float(model.index[0]),
    curand_uniform(randState)*float(model.index[1])
  };

  // Iterate over a Number of Steps

  size_t find = model.index.flatten(pos);
  float height = model.height[find]*scale.y;
  vec2 speed = steepest(model, pos);
  float stable = stable_height(model, pos, param, height, speed);
  float excess = height - stable;
  
  if(excess < 0.0f)
    return;

  // Compute Mass Initial Condition
  
  float mass = param.settling * excess;
  atomicAdd(&model.height[find], - P * param.settling * excess / scale.y);

  for(size_t age = 0; age < 128; ++age){

    pos += speed;
    if(model.index.oob(pos)){
      return;
    }

    size_t find = model.index.flatten(pos);
    float height = model.height[find]*scale.y;
    vec2 speed = steepest(model, pos);
    float stable = stable_height(model, pos, param, height, speed);
    float excess = height - stable;
    if(excess < 0.0f){
      excess = -glm::min(-excess, mass);
    }

    atomicAdd(&model.height[find], - P * param.settling * excess / scale.y);
    mass += param.settling * excess;

    if(mass == 0.0f)
      break;

  }

}
*/

__global__ void debris_flow(model_t model, const size_t N, const param_t param){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= model.elem) return;

  // Parameters

  const vec3 scale = model.scale;
  const float g = param.gravity;

  // Spawn Particle at Random Position

  curandState* randState = &model.rand[ind];
  const float P = float(model.elem)/float(N); // Sample Probability
  vec2 pos = vec2{
    curand_uniform(randState)*float(model.index[0]),
    curand_uniform(randState)*float(model.index[1])
  };

  if(model.index.oob(pos))
    return;

  // Iterate over a Number of Steps

  float mass = 0.0f;  // Currently Transported Mass

  for(size_t age = 0; age < 128; ++age){

    // Compute Height

    lerp5_t<float> lerp;
    lerp.gather(model.height, model.index, ivec2(pos));
    const vec2 grad = lerp.grad(model.scale);
    const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
    vec2 speed = g * vec2(normal.x, normal.y);


    vec2 npos = pos;
    if(glm::length(speed) < 0.00001f)
      return;
    else npos += sqrt(2.0f) * glm::normalize(speed);

    if(model.index.oob(npos)){
      return;
    }

    // Note: Effective Horizontal Distance Traveled!
    //  We scale this by the actual physical length.
    float dist = sqrt(2.0f) * glm::length(vec2(scale.x, scale.z));

    // Compute Height-Difference
    const size_t i = model.index.flatten(pos);
    float h = model.height[i]*scale.y;
    float h_next = model.height[model.index.flatten(npos)]*scale.y;

    // Note: This slope stability function can be replaced
    //  with something more complex in general!
    // The stable height becomes the lowst stable height

    // We compute the difference to the stable amount
    float stable = h_next + param.maxdiff*dist;
    float excess = h - stable;

    // if excess is less than zero,
    //  that means that we can add mass if we have any!
    if(excess < 0.0f){
      float transfer = glm::min(-excess, mass);
      int find = model.index.flatten(pos);
      atomicAdd(&model.height[find], param.settling * transfer / scale.y);
      mass -= param.settling * transfer;
    }

    // if it is equal zero, we are exactly stable

    else if(excess > 0.0f){

      float transfer = excess;
      int find = model.index.flatten(pos);
      atomicAdd(&model.height[find], -param.settling * transfer / scale.y);
      mass += param.settling * transfer;
    }

    if(mass == 0.0f)
      break;

    pos = npos;

  }

//  if(!model.index.oob(pos)){
//    const size_t i = model.index.flatten(pos);
//    atomicAdd(&model.height[i], mass);
//  }

}


} // end of namespace soil

#endif