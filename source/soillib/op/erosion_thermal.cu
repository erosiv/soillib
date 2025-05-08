#ifndef SOILLIB_OP_EROSION_THERMAL_CU
#define SOILLIB_OP_EROSION_THERMAL_CU
#define HAS_CUDA

#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <soillib/op/erosion.hpp>
#include <soillib/op/gather.hpp>

#include <soillib/op/erosion_map.cu>

//! Thermal Erosion / Debris-Flow Particle
//!
//! Debris-Flow Mass Estimator for Mass-Transport 
//! Based on Bank Stability, Abrasion and Transport

namespace soil {

//! Debris-Flow Particle
struct debris_t {

  vec2 pos;     //!< World Position [pix]
  vec2 speed;   //!< World Velocity [m/s]
  float Q;      //!< Weighted Sampling Probability
  int ind;      //!< Nearest Support Index

  vec2 dspeed;  //!< Velocity Rate    [m/s^2]
  float mass;   //!< Debris Mass Rate [kg/s]

};

namespace debris {

//
// Mass-Transfer Model
//

__device__ float deposit(const param_t& param, const float mass) {

  const float kds = param.settleRate;
  return kds * mass;

}

__device__ float suspend(const model_t& model, const param_t& param, const float hdiff) {

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]
  
  const float kth0 = param.thermalRate;
  const float suspend = kth0 * glm::max(0.0f, hdiff) * Ac;
  return suspend;

}

__device__ float limit(float transfer, const float mass, const float hdiff, const vec3 scale){

  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

//  if(transfer > 0.0f){
//    const float maxtransfer = 0.57f * glm::length(cl) * Ac;
//    transfer = transfer * glm::max(1.0f, maxtransfer / transfer);
//  }

  //   
  // if(transfer < 0.0f){
  //   const float maxtransfer = 0.15f * glm::max(0.0f, hdiff) * Ac;
  //   transfer = -glm::min(-transfer, maxtransfer);
  // }

  transfer = glm::min(transfer, mass);
  return transfer;

}

//
// Solution Loop Functions
//

//! Initialize Particle Data from Model
__device__ void init(debris_t& part, model_t& model, const param_t& param){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float& R = param.rainfall;        // Rainfall Amount  [m/y]
  const float& g = param.gravity;         // Specific Gravity [m/s^2]
  const float& nu = param.viscosity;      // Kinematic Viscosity [m^2/s]

  const float dt = param.timeStep;

  const vec2 normal = __steepest(model, part.pos, scale);
  part.speed = g * normal;
  part.dspeed = part.speed; //!< Velocity Rate [m^2/s^2]

  float hdiff = __hdiff(model, param, part.pos);
  float suspend = debris::suspend(model, param, hdiff);
  part.mass = suspend;

}

__device__ void move(const model_t& model, debris_t& part){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  if(glm::length(part.speed) == 0.0f)
    return;

  const float ds = glm::length(cl)/glm::length(part.speed);
  part.pos = part.pos + ds * (part.speed / cl);
  part.ind = model.index.flatten(part.pos);

}

//! Integrate Sub-Solution Quantities in Quasi-Static Time
__device__ void integrate(const model_t& model, const param_t& param, debris_t& part){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  
  const float g = param.gravity;      // Specific Gravity [m/s^2]
  const float k1 = param.debrisShear; // Shear-Stress Bed-Shear
  const float k2 = 0.0f;              // Shear-Stress Viscosity

  const float ds = glm::length(cl)/glm::length(part.speed);
  
  //! Explicit Euler Forward Integration for Gravity
  const vec2 normal = __steepest(model, part.pos, scale);
  part.speed = part.speed + ds * g * normal;
  part.speed =  1.0f/(1.0f + ds * (k1+k2))*part.speed;// + ds*k2/(1.0f + ds*(k1+k2))*average_speed;
  part.dspeed = 1.0f/(1.0f + ds * (k1+k2))*part.dspeed;

}

//! Mass-Transfer Characteristic Integration
__device__ void integrate_mt(const model_t& model, const param_t& param, debris_t& part){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  float deposit = debris::deposit(param, part.mass);
  deposit = glm::min(deposit, part.mass);
  part.mass -= deposit;

}

//! Debris-Flow Estimate Accumulation
__device__ void track(model_t& model, const debris_t& part){

  atomicAdd(&model.debris_track[part.ind], (part.mass)/part.Q);
  atomicAdd(&model.debris_momentum_track[part.ind].x, (part.mass*part.dspeed.x)/part.Q);
  atomicAdd(&model.debris_momentum_track[part.ind].y, (part.mass*part.dspeed.y)/part.Q);

}


//
// Kernels
//

__global__ void solve(model_t model, const size_t N, const param_t param) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) 
    return;

  debris_t part;                    //!< Data along Trajectory / Per-Particle
  __sample(part, model, n, N);      //!< Sample the Trajectory
  debris::init(part, model, param); //!< Initialize Particle Properties

  // Note: Parameterize
  for(size_t age = 0; age < 256; ++age) {

    debris::track(model, part);   //!< Accumulate Estimate
    debris::move(model, part);    //!< Move Trajectory
    if(model.index.oob(part.pos))
      break;
    
    debris::integrate_mt(model, param, part); //!< Integrate Mass-Transfer
    debris::integrate(model, param, part);    //!< Integrate Trajectory
    if(glm::length(part.speed) == 0.0f)
      break;
  
  }

}

__global__ void mt(model_t model, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= model.height.elem())
    return;

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  const float mass = model.debris[n];               // Suspended Mass Function
  const vec2 pos = model.index.unflatten(n);
  const float hdiff = __hdiff(model, param, pos + vec2(0.5f));

//  const float dt = param.timeStep;
  const float deposit = debris::deposit(param, mass);
  const float suspend = debris::suspend(model, param, hdiff);
  float transfer = (deposit - suspend);
  transfer = debris::limit(transfer, mass, hdiff, scale);
  __transfer(model, n, transfer, Z);

}

}

} // end of namespace soil

#endif