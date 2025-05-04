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
__device__ vec2 steepest_speed(const model_t& model, const param_t param, const vec2 pos) {

  const vec2 shift[8] = {
    vec2(-1.0, -1.0),
    vec2( 0.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  0.0),
    vec2( 1.0,  0.0),
    vec2(-1.0,  1.0),
    vec2( 0.0,  1.0),
    vec2( 1.0,  1.0)
  };

  int mini = -1;
  float minh = model.height[model.index.flatten(pos)];

  for(int i = 0; i < 8; ++i){
    vec2 npos = pos + shift[i];
    if(model.index.oob(npos)){
      continue;
    }
    float h = model.height[model.index.flatten(npos)];
    if(h < minh){
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
  //lerp.gather(model.height, model.index, pos);
  const vec2 grad = lerp.grad(model.scale);
  const vec3 normal = glm::normalize(vec3(-grad.x, -grad.y, 1.0f));
  return g * vec2(normal.x, normal.y);
  */
  
}

__device__ float _transfer(float* buf, float val, const float max){
  if(abs(val) > 1E-8){
    val = val * glm::min(1.0f, max/abs(val)); // Cap Val at Max
    atomicAdd(buf, val);                      // Transfer Val
  }
  return val;                               // Return Value
}

struct debris_t {

  vec2 pos;   //!< World Position [pix]
  vec2 speed; //!< World Velocity [m/s]

  float Q;    //!< Weighted Sampling Probability
  int ind;    //!< Nearest Support Index

  float mass; //!< Debris Mass

};

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

  part.speed = steepest_speed(model, param, part.pos);
  part.mass = 0.0f;

}

__device__ void __track(model_t& model, const debris_t& part){

  // Note: Place the debris-flow mass tracking here...

//  atomicAdd(&model.mass_track[part.ind], (part.sed)/part.Q);
//  atomicAdd(&model.discharge_track[part.ind], (part.vol)/part.Q);
//  atomicAdd(&model.momentum_track[part.ind].x, (part.vol*part.dspeed.x)/part.Q);
//  atomicAdd(&model.momentum_track[part.ind].y, (part.vol*part.dspeed.y)/part.Q);

}

__device__ void __move(const model_t& model, debris_t& part){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  const float ds = glm::length(cl)/glm::length(part.speed);
  part.pos = part.pos + ds * (part.speed / cl);
  part.ind = model.index.flatten(part.pos);

}

//! Integrate Sub-Solution Quantities in Quasi-Static Time
__device__ void __integrate(const model_t& model, const param_t& param, debris_t& part){

  part.speed = steepest_speed(model, param, part.pos);

}

//! Note: This potentially removes a lot of mass at once.
//! we need to make sure that we are limiting correctly!
//!
__device__ void __integrate_mt(model_t& model, const param_t& param, debris_t& part){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  vec2 npos = part.pos + glm::normalize(part.speed);
  if(model.index.oob(npos)){
    return;
  }

  int find = part.ind;
  int nind = model.index.flatten(npos);

  const float kds = param.settleRate;
  const float deposit = kds * part.mass;

  // Compute Equilibrium Mass Transfer

  // Note: Because of the way the height-lookup works, we are
  //  doing a floor of the position here. If the position was
  //  sampled smoothly, this would not be necessary.
  // const float dist = glm::length(cl*vec2(ivec2(npos) - ivec2(pos)));
  const float dist = glm::length(cl*(npos-part.pos));
  part.pos = npos;

  // Stable Bank-Height Computation:

  float hf_0 = scale.z * model.height[find];
  float hn_0 = scale.z * model.height[nind];

  // for some reason, this is making the sediment buffer negative... not good.
  //  this needs to be reconsidered in terms of overall stability.
  float hf_1 = glm::max(0.0f, scale.z * model.sediment[find]);
  float hn_1 = glm::max(0.0f, scale.z * model.sediment[nind]);
  float hf = (hf_0 + hf_1);
  float hn = (hn_0 + hn_1);

  const float stable1 = (hn + param.critSlope*dist);  // [m]
  const float stable0 = (hn + param.critSlope*dist);  // [m]
  
  const float kth0 = param.thermalRate;
  const float suspend = - kth0 * glm::max(0.0f, hf - stable1) * Ac;

  const float dt = param.timeStep;
  float transfer = dt * (deposit + suspend);

  // Limit

  if(transfer > 0.0f){
    const float maxtransfer = 0.05f * glm::max(0.0f, stable1 - hf) * Ac * part.Q;
    transfer = glm::min(transfer, maxtransfer);
  }
    
  else if(transfer < 0.0f){
    const float maxtransfer = 0.05f * glm::max(0.0f, hf - stable1) * Ac * part.Q;
    transfer = -glm::min(-transfer, maxtransfer);
  }

  transfer = glm::min(transfer, part.mass);

  // Single Material
  atomicAdd(&model.height[find], transfer / part.Q / Z);
  part.mass -= transfer;

//    // Multi-Material
//    if(transfer > 0.0f){ // Add Material to Map
//    
//    const float maxtransfer = glm::max(0.0f, stable1 - hf) * Ac * part.Q;
//    transfer = glm::min(transfer, maxtransfer);
//    transfer = glm::min(transfer, part.mass);
//    transfer = glm::max(0.0f, transfer);
//    
//    atomicAdd(&model.sediment[find], transfer / part.Q / Z);
//    part.mass -= transfer;
//    
//  }
//  
//  else { // Remove Material from Map
//  
//  const float maxtransfer = glm::max(0.0f, hf - stable1) * Ac * part.Q;
//  transfer = -glm::min(-transfer, maxtransfer);
//  
//  const float maxt1 = hf_1 * Ac * part.Q;
//  float t1 = transfer * glm::min(1.0f, glm::abs(maxt1/transfer));
//  
//  atomicAdd(&model.sediment[find], t1 / part.Q / Z);
//  part.mass -= t1;
//  
//  transfer -= t1;
//  atomicAdd(&model.height[find], transfer / part.Q / Z );
//  part.mass -= transfer;
//  
//  }

}

__global__ void solve_debris(model_t model, const size_t N, const param_t param) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) 
    return;

  debris_t part;                //!< Data along Trajectory / Per-Particle
  __sample(part, model, n, N);  //!< Sample the Trajectory
  __init(part, model, param);   //!< Initialize Particle Properties
  __track(model, part);

  int past[2] = {-1, -1};
  past[1] = part.ind;
  const int maxloop = 1;
  int nloop = 0;

  // Note: Parameterize
  for(size_t age = 0; age < 256; ++age) {

    __integrate_mt(model, param, part); //!< Integrate Mass-Transfer
    __move(model, part);                //!< Move Trajectory
    if(model.index.oob(part.pos))
      break;

    __integrate(model, param, part);    //!< Integrate Trajectory
    __track(model, part);

    // Short Loop Detection...
    if(part.ind == past[0]) ++nloop;
    if(part.ind == past[1]) ++nloop;
    if(nloop >= maxloop) break;

    past[0] = past[1];
    past[1] = part.ind;

    if(glm::length(part.speed) == 0.0f)
      break;
  
  }

}

} // end of namespace soil

#endif