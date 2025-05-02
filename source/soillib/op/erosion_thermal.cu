#ifndef SOILLIB_OP_EROSION_THERMAL_CU
#define SOILLIB_OP_EROSION_THERMAL_CU
#define HAS_CUDA

#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <soillib/op/erosion.hpp>
#include <soillib/op/gather.hpp>

namespace soil {

//! Fluvial Erosion Mass-Transfer System
//! Single-Material
//!
// __device__ void __mt1(model_t& model, particle_t& part, const param_t& param, const size_t N){
// 
//   const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
//   const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
//   const float Ac = scale.x*scale.y;       // Cell Area [m^2]
//   const float Z = Ac * scale.z;           // Height Conversion [m^3]
// 
//   const float dt = param.timeStep;        // Geological Timestep [y]
//   const float kd = param.depositionRate;  // Fluvial Deposition Rate [1/y]
//   const float ks = param.suspensionRate;  // Fluvial Suspension Rate [(m^3/y)^-0.4]
// 
//   float discharge = model.discharge[part.ind];  // Discharge Function
//   float slope = __slope(model, part, param);    // Slope Function
//   float alpha = (slope < 0.0f)?1.0f:0.0f;       // Activation Function
//   float suspend = dt * ks * part.vol * slope * alpha * pow(discharge, 0.4f); // [kg]
//   float deposit = dt * kd * part.sed;                                        // [kg]
// 
//   // Single Material, Implicit Euler Scheme
//   //  This use an activation function which lowers the amount transferred
//   //  which scales with the amount of equilibriation force. Note that this
//   //  tends to over-damp, which is why we don't use it.
// 
// //    float kq = ks * part.vol * alpha * pow(discharge, 0.4f) / glm::length(cl);
// //    float transfer = 1.0f / (1.0f + dt * kq) * (suspend + deposit);
// //    atomicAdd(&model.height[part.ind], transfer / Z / Q);
// //    part.sed -= transfer;
// 
//   // Single Material, Explicit Euler Scheme
//   //  This use an activation function (maxtransfer), which limits the
//   //  total amount of mass that can be moved based on the slope.
//   //  Similar to the implicit scheme, which uses a similar construction
//   //  but that scales with the rate.
// 
//   // Note: Maxtransfer here is damped for stability. This should be
//   //  attempted to be removed using alternative stabilizing methods.
//   float transfer = (deposit + suspend);
//   const float maxtransfer = 0.1f * slope * glm::length(cl) / scale.z * Z * part.Q;
//   const float tmin = transfer * glm::min(1.0f, glm::abs(maxtransfer/transfer));
//   const float tmax = part.sed;
//   transfer = glm::clamp(transfer, tmin, tmax);
// 
//   atomicAdd(&model.height[part.ind], transfer / Z / part.Q);
//   part.sed -= transfer;
// 
// }

//! Fluvial Erosion Mass-Transfer System
//! Multi-Material
//!
// __device__ void __mt2(model_t& model, particle_t& part, const param_t& param, const size_t N){
// 
//   const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
//   const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
//   const float Ac = scale.x*scale.y;       // Cell Area [m^2]
//   const float Z = Ac * scale.z;           // Height Conversion [m^3]
// 
//   const float dt = param.timeStep;        // Geological Timestep [y]
//   const float kd = param.depositionRate;  // Fluvial Deposition Rate [1/y]
//   const float ks = param.suspensionRate;  // Fluvial Suspension Rate [(m^3/y)^-0.4]
// 
//   //
//   // Mass-Transfer
//   //  Compute Equilibrium Mass from Slope and Discharge
//   //  Transfer Mass and Scale by Sampling Probability
//   
//   float discharge = model.discharge[part.ind];  // Discharge Function
//   float slope = __slope(model, part, param);    // Slope Function
//   float alpha = (slope < 0.0f)?1.0f:0.0f;       // Activation Function
//   float suspend = dt * ks * part.vol * slope * alpha * pow(discharge, 0.4f); // [kg]
//   float deposit = dt * kd * part.sed;                                        // [kg]
// 
//   // Multi-Material Mass Transfer
//   float transfer = (deposit + suspend);
//   const float maxtransfer = 0.1f * slope * glm::length(cl) / scale.z * Z * part.Q;
//   const float tmin = transfer * glm::min(1.0f, glm::abs(maxtransfer/suspend));
//   const float tmax = part.sed;
//   transfer = glm::clamp(transfer, tmin, tmax);
// 
//   if(transfer > 0.0f){  // Add Material to Map (Note: Single Material Model)
// 
//     atomicAdd(&model.sediment[part.ind], transfer / Z / part.Q);
//     part.sed -= transfer;
// 
//   }
// 
//   else if(transfer < 0.0f){ // Remove Sediment from Map
// 
//     const float maxtransfer = 0.1f * model.sediment[part.ind] * Z * part.Q;
//     float t1 = transfer * glm::min(1.0f, glm::abs(maxtransfer/transfer));
//     atomicAdd(&model.sediment[part.ind], t1 / Z / part.Q);
//     part.sed -= t1;
// 
//     transfer -= t1;
//     atomicAdd(&model.height[part.ind], transfer / Z / part.Q);
//     part.sed -= transfer;
// 
//   }
// 
// }

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