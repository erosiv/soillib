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
__device__ vec2 steepest_speed(model_t& model, const param_t param, const ivec2 pos) {

  const vec3 scale = model.scale;
  const float g = param.gravity;

  lerp5_t<float> lerp;
  lerp.gather(model.height, model.index, pos);
  const vec2 grad = lerp.grad(model.scale);
  const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
  return g * vec2(normal.x, normal.y);

}

//! Debris Flow Kernel Implementation
//!
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

  // Note: Parameterize
  for(size_t age = 0; age < 1024; ++age){

    // Motion Along Characteristic

    vec2 speed = steepest_speed(model, param, pos);
    if(glm::length(speed) < 0.00001f){
      return;
    }

    vec2 npos = pos + sqrt(2.0f) * glm::normalize(speed);
    if(model.index.oob(npos)){
      return;
    }

    // Compute Equilibrium Mass Transfer

    int find = model.index.flatten(pos);
    int nind = model.index.flatten(npos);

    // Stable Bank-Height Computation:
    // This can be replaced with a more complex expression if desired.
    //  For instance, it can include the discharge function.

    float h = model.height[find]*scale.y;
    float h_next = model.height[nind]*scale.y;
    float dist = sqrt(2.0f) * glm::length(vec2(scale.x, scale.z));
    float stable = h_next + param.maxdiff*dist;
    
    float excess = h - stable;
    if(excess < 0.0f){
      excess = -glm::min(-excess, mass);
    }

    atomicAdd(&model.height[find], - param.settling * excess / scale.y);
    mass += param.settling * excess;
    if(mass == 0.0f)
      break;

    pos = npos;

  }

}


} // end of namespace soil

#endif