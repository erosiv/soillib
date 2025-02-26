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

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m]
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float g = param.gravity;
  const float dt = param.timeStep;

  float mass = 0.0f;  // Currently Transported Mass

  // Spawn Particle at Random Position

  curandState* randState = &model.rand[ind];
  vec2 pos = vec2{
    curand_uniform(randState)*float(model.index[0]),
    curand_uniform(randState)*float(model.index[1])
  };
  const float P = 1.0f / float(model.index.elem());
  const float Q = P * float(N); // Sampling Probability Scale

  // Iterate over a Number of Steps

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

    // Note: Because of the way the height-lookup works, we are
    //  doing a floor of the position here. If the position was
    //  sampled smoothly, this would not be necessary.
    // const float dist = glm::length(cl*vec2(ivec2(npos) - ivec2(pos)));
    const float dist = glm::length(cl*(npos-pos));
    pos = npos;

    // Stable Bank-Height Computation:

    float hf_0 = scale.z * model.height[find];
    float hn_0 = scale.z * model.height[nind];

    // for some reason, this is making the sediment buffer negative... not good.
    //  this needs to be reconsidered in terms of overall stability.
    float hf_1 = glm::max(0.0f, scale.z * model.sediment[find]);
    float hn_1 = glm::max(0.0f, scale.z * model.sediment[nind]);
    float hf = (hf_0 + hf_1);
    float hn = (hn_0 + hn_1);

    const float kds =  param.settleRate;
    const float kth1 = param.thermalRate;
    const float kth0 = param.thermalRate;

    const float stable1 = (hn + param.critSlope*dist);  // [m]
    const float stable0 = (hn + param.critSlope*dist);  // [m]

    const float deposit =  dt * kds * mass;
    const float suspend = -dt * kth1 * glm::max(0.0f, hf - stable1) * Ac;
    float transfer = (deposit + suspend);
    if(transfer == 0.0f)
      continue;

    /*
    // Single Material
    if(transfer > 0.0f){

      const float maxtransfer = glm::max(0.0f, stable1 - hf) * Ac * Q;
      transfer = glm::min(transfer, maxtransfer);
      transfer = glm::min(transfer, mass);

      atomicAdd(&model.height[find], transfer / Q / scale.z / Ac);
      mass -= transfer;

    }

    else if(transfer < 0.0f){

      const float maxtransfer = glm::max(0.0f, hf - stable1) * Ac * Q;
      transfer = -glm::min(-transfer, maxtransfer);

      atomicAdd(&model.height[find], transfer / Q / scale.z / Ac);
      mass -= transfer;

    }
    */

    // Multi-Material
    if(transfer > 0.0f){ // Add Material to Map

      const float maxtransfer = glm::max(0.0f, stable1 - hf) * Ac * Q;
      transfer = glm::min(transfer, maxtransfer);
      transfer = glm::min(transfer, mass);
      transfer = glm::max(0.0f, transfer);

      atomicAdd(&model.sediment[find], transfer / Q / scale.z / Ac);
      mass -= transfer;

    }

    else { // Remove Material from Map

      const float maxtransfer = glm::max(0.0f, hf - stable1) * Ac * Q;
      transfer = -glm::min(-transfer, maxtransfer);

      const float maxt1 = hf_1 * Ac * Q;
      float t1 = transfer * glm::min(1.0f, glm::abs(maxt1/transfer));

      atomicAdd(&model.sediment[find], t1 / Q / scale.z / Ac);
      mass -= t1;

      transfer -= t1;
      atomicAdd(&model.height[find], transfer / Q / Ac / scale.z );
      mass -= transfer;

    }

  }

}

} // end of namespace soil

#endif