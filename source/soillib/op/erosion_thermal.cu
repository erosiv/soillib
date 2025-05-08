#ifndef SOILLIB_OP_EROSION_THERMAL_CU
#define SOILLIB_OP_EROSION_THERMAL_CU
#define HAS_CUDA

#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <soillib/op/erosion.hpp>
#include <soillib/op/gather.hpp>

namespace soil {

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
__device__ vec2 steepest_speed(const model_t& model, const param_t param, const ivec2 pos) {

  const ivec2 shift[8] = {
    ivec2(-1.0, -1.0),
    ivec2( 0.0, -1.0),
    ivec2( 1.0, -1.0),
    ivec2(-1.0,  0.0),
    ivec2( 1.0,  0.0),
    ivec2(-1.0,  1.0),
    ivec2( 0.0,  1.0),
    ivec2( 1.0,  1.0)
  };

  int mini = -1;
  float minh = model.height[model.index.flatten(pos)] + model.sediment[model.index.flatten(pos)];;
  
  for(int i = 0; i < 8; ++i){
    ivec2 npos = pos + shift[i];
    if(model.index.oob(npos)){
      continue;
    }
    float h = model.height[model.index.flatten(npos)] + model.sediment[model.index.flatten(npos)];
    if(h <= minh){
      mini = i;
      minh = h;
    }
  }

  if(mini == -1)
    return vec2(0.0f);
  else return shift[mini];

    
  /*
  const vec3 scale = model.scale;
  const float g = param.gravity;
  
  lerp5_t<float> lerp;
  lerp.gather(model.height, model.sediment, model.index, pos);
  const vec2 grad = lerp.grad(model.scale);
  const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
  return g * vec2(normal.x, normal.y);
  */
  
}

struct debris_t {

  vec2 pos;   //!< World Position [pix]
  vec2 speed; //!< World Velocity [m/s]
  vec2 dspeed;  //!< Velocity Rate

  float Q;    //!< Weighted Sampling Probability
  int ind;    //!< Nearest Support Index

  float mass; //!< Debris Mass

};

//
// Mass-Transfer Functions
//

__device__ float __hdiff(const model_t& model, const param_t& param, const ivec2 pos) {

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  const vec2 dir = steepest_speed(model, param, pos);
  vec2 npos = vec2(pos) + dir;
  if(model.index.oob(npos)){
    return 0.0f;
  }

  const float dist = glm::length(cl*dir);

  // Stable Bank-Height Computation:

  int find = model.index.flatten(pos);
  int nind = model.index.flatten(npos);

  const float hf_0 = model.height[find];
  const float hn_0 = model.height[nind];
  const float hf_1 = model.sediment[find];
  const float hn_1 = model.sediment[nind];

  const float hf = scale.z * (hf_0 + hf_1);
  const float hn = scale.z * (hn_0 + hn_1);

  const float stable = (hn + param.critSlope*dist);  // [m]
  const float hdiff = hf - stable;
  return hdiff;

}

__device__ float __deposit_debris(const param_t& param, const float mass) {
  const float kds = param.settleRate;
  return kds * mass;
}

// Compute Equilibrium Mass Transfer

// Note: Because of the way the height-lookup works, we are
//  doing a floor of the position here. If the position was
//  sampled smoothly, this would not be necessary.
// const float dist = glm::length(cl*vec2(ivec2(npos) - ivec2(pos)));
__device__ float __suspend_debris(const model_t& model, const param_t& param, const float hdiff) {

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]
  
  const float kth0 = param.thermalRate;
  const float suspend = kth0 * glm::max(0.0f, hdiff) * Ac;
  return suspend;

}

__device__ float __limit_debris(float transfer, const float mass, const float hdiff, const vec3 scale){

  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]
  
  // if(transfer > 0.0f){
  //   const float maxtransfer = 0.5f * glm::max(0.0f, -hdiff) * Ac;
  //   transfer = glm::min(transfer, maxtransfer);
  // }
  //   
  // if(transfer < 0.0f){
  //   const float maxtransfer = 0.15f * glm::max(0.0f, hdiff) * Ac;
  //   transfer = -glm::min(-transfer, maxtransfer);
  // }

  transfer = glm::min(transfer, mass);
  return transfer;

}

//
// Core Functions
//

__device__ void __sample(debris_t& part, model_t& model, const size_t n, const size_t N){

  part.pos = vec2 {
    curand_uniform(&model.rand[n])*float(model.index[0]),
    curand_uniform(&model.rand[n])*float(model.index[1])
  };
  part.ind = model.index.flatten(part.pos);

  const float P = 1.0f / float(model.index.elem()); // Sampling Probability
  part.Q = P * float(N);                            // Sampling Weight

}

//! Initialize Particle Data from Model
//!
__device__ void __init(debris_t& part, model_t& model, const param_t& param){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float& R = param.rainfall;        // Rainfall Amount  [m/y]
  const float& g = param.gravity;         // Specific Gravity [m/s^2]
  const float& nu = param.viscosity;      // Kinematic Viscosity [m^2/s]

  const float dt = param.timeStep;
  part.speed = steepest_speed(model, param, part.pos);
  part.dspeed = part.speed; //!< Velocity Rate [m^2/s^2]

  float hdiff = __hdiff(model, param, part.pos);
  float suspend = __suspend_debris(model, param, hdiff);
  part.mass = suspend;

}

//! Debris-Flow Estimate Accumulation
__device__ void __track(model_t& model, const debris_t& part){

  atomicAdd(&model.debris_track[part.ind], (part.mass)/part.Q);
  atomicAdd(&model.debris_momentum_track[part.ind].x, (part.mass*part.dspeed.x)/part.Q);
  atomicAdd(&model.debris_momentum_track[part.ind].y, (part.mass*part.dspeed.y)/part.Q);

}

__device__ void __move(const model_t& model, debris_t& part){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  if(glm::length(part.speed) == 0.0f)
    return;

  const float ds = glm::length(cl)/glm::length(part.speed);
  part.pos = part.pos + ds * (part.speed / cl);
  part.ind = model.index.flatten(part.pos);

}

//! Integrate Sub-Solution Quantities in Quasi-Static Time
__device__ void __integrate(const model_t& model, const param_t& param, debris_t& part){

  part.speed = steepest_speed(model, param, part.pos);
  part.dspeed = part.speed; //!< Velocity Rate [m^2/s^2]

}

//! Mass-Transfer Characteristic Integration
//!
//! Note that the particle-mass is technically a mass-rate,
//! 
__device__ void __integrate_mt(model_t& model, const param_t& param, debris_t& part){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  float deposit = __deposit_debris(param, part.mass);
  deposit = glm::min(deposit, part.mass);
  part.mass -= deposit;

}

__global__ void mt_debris(model_t model, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= model.height.elem())
    return;

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  const vec2 pos = model.index.unflatten(n);
  const float mass = model.debris[n];               // Suspended Mass Function
  const float hdiff = __hdiff(model, param, pos);

//  const float dt = param.timeStep;
  const float deposit = __deposit_debris(param, mass);
  const float suspend = __suspend_debris(model, param, hdiff);
  float transfer = (deposit - suspend);
  transfer = __limit_debris(transfer, mass, hdiff, scale);

  // Single-Material Mass-Transfer
//  model.height[n] += transfer / Z;

  // Multi-Material Mass-Transfer
  if(transfer >= 0.0f){

    model.sediment[n] += transfer / Z;

  } else {

    const float maxtransfer = model.sediment[n] * Z;
    float t1 = transfer * glm::min(1.0f, glm::abs(maxtransfer/transfer));
    model.sediment[n] += t1 / Z;

    transfer -= t1;
    model.height[n] += transfer / Z;

  }

}

__global__ void solve_debris(model_t model, const size_t N, const param_t param) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) 
    return;

  debris_t part;                //!< Data along Trajectory / Per-Particle
  __sample(part, model, n, N);  //!< Sample the Trajectory
  __init(part, model, param);   //!< Initialize Particle Properties

  // Note: Parameterize
  for(size_t age = 0; age < 256; ++age) {

    __track(model, part);         //!< Accumulate Estimate
    __move(model, part);          //!< Move Trajectory
    if(model.index.oob(part.pos))
      break;
    
    __integrate_mt(model, param, part); //!< Integrate Mass-Transfer
    __integrate(model, param, part);    //!< Integrate Trajectory

//    if(glm::length(part.speed) == 0.0f)
//      break;
  
  }

}

} // end of namespace soil

#endif