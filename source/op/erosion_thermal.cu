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
  lerp.gather(model.height, model.sediment, model.index, pos);
  //lerp.gather(model.height, model.index, pos);
  const vec2 grad = lerp.grad(model.scale);
  const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
  return g * vec2(normal.x, normal.y);

}

__device__ float _transfer(float* buf, float val, const float max){
  if(abs(val) > 1E-8){
    val = val * glm::min(1.0f, max/abs(val)); // Cap Val at Max
    atomicAdd(buf, val);                      // Transfer Val
  }
  return val;                               // Return Value
}

//! Debris Flow Kernel Implementation
//!
//! Utilizes a stable bank height to compute the eroded material.
//! This can use a more complex expression if desired, for instance
//! incorporating the discharge function / agitation / lift vs gravity
//! vs friction coefficient per material, etc.
//!
//! The computation occurs in scale-free dimensions.
//!
//! Currently, this uses a simple explicit integration which requires
//! limiting the parameters to a maximum value. At high resolutions,
//! this causes some stability issues due to the sampling scaling.
//! Using an implicit method would solve this problem directly.
//!
__global__ void debris_flow(model_t model, const size_t N, const param_t param){

  const unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind >= model.elem) return;

  // Parameters

  const vec3 scale = model.scale;
  const float g = param.gravity;

  // Spawn Particle at Random Position

  curandState* randState = &model.rand[ind];
  vec2 pos = vec2{
    curand_uniform(randState)*float(model.index[0]),
    curand_uniform(randState)*float(model.index[1])
  };
  const float P = 1.0f / float(model.index.elem());

  // Iterate over a Number of Steps

  float mass = 0.0f;  // Currently Transported Mass

  // Note: Parameterize
  for(size_t age = 0; age < 256; ++age){

    // Motion Along Characteristic

    vec2 npos = pos;

    vec2 speed = steepest_speed(model, param, pos);
    if(glm::length(speed) > 0.0f){
      npos = pos + glm::normalize(speed);
    }

    if(model.index.oob(npos)){
      return;
    }

    // Compute Equilibrium Mass Transfer

    int find = model.index.flatten(pos);
    int nind = model.index.flatten(npos);

    // Stable Bank-Height Computation:

    float hf_0 = model.height[find];
    float hf_1 = model.sediment[find];
    float hn_0 = model.height[nind];
    float hn_1 = model.sediment[nind];
    float hf = (hf_0 + hf_1);
    float hn = (hn_0 + hn_1);

    float dist = glm::length(vec2(scale.x, scale.z)*(npos - pos));
    float stable = (hn + param.maxdiff*dist/scale.y);
    
    // Arbitrary Rate Limiting due to Explicit Method:
    float kth = glm::min(0.8f, param.settling  / P / float(N));

    float suspend = -kth * glm::max(0.0f, hf - stable);
    float deposit =  kth * mass;

    // Deposit Mass onto Sediment Field, Limited by Suspended Mass
    const float t1 = _transfer(&model.sediment[find], deposit, mass);
    mass -= t1;

    // Suspend Mass from Sediment Field, Limited by Total Height of Field
    const float t2 = _transfer(&model.sediment[find], suspend, hf_1 + t1);
    mass -= t2;

    // Suspend Mass from Bedrock Field, Unlimited Amount
    const float t3 = _transfer(&model.height[find], suspend - t2, INFINITY);
    mass -= t3;

    pos = npos;

  }

}


} // end of namespace soil

#endif