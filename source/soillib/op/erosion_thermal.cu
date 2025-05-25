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

//! Suspension Rate [m/s]
__device__ float landslide_suspend(const param_t& param, const float slope) {

  const float g = param.gravity;            //!< Specific Gravity [m/s^2]
  const float kdl = param.debrisCreepRate;  //!< Landslide Erosion Rate [1/s]
  const float rho = param.debrisDensity;    //!< Debris Density [kg / m^3]
  const float theta = param.critSlope;

  const float diff = glm::max(0.0f, slope-theta);
  return kdl * g * diff;

}

//! Deposition Rate [m^3/s]
__device__ float deposit(const param_t& param, const float dt, const float mass, const vec2 momentum, const vec3 scale) {

  const float kdd = param.debrisDepositionRate; //!< Thermal Deposition Rate [1/s]
  const float decay = (1.0f-__expf(-dt * kdd));      //!< Total Decay Factor []
  return decay * mass;

}

//! Suspension Rate [m/s]
__device__ float suspend(const param_t& param, const float mass, const float slope, const vec2 momentum) {

  const float kds = param.debrisSuspensionRate; //!< Thermal Deposition Rate [1/s]
  const float rho = param.debrisDensity;        //!< Debris Density
  const float g = param.gravity;                //!< Specific Gravity [m/s^2]
  const float tau = param.debrisYieldStress;    //!< Debris-Flow Bed Shear Pa s
  const float mu = param.debrisViscosity;       //!< Debris Flow Viscosity
  const float theta = param.critSlope;          //!< Debris Critical Slope

  const vec2 speed = (mass > 1.0f)?momentum / mass : vec2(0.0f);
  const float stress_gravity = g * mass * glm::max(slope - theta, 0.0f);
  const float stress_yield = tau / rho;
  const float stress_viscous =  mu * glm::length(speed);// / mass;
  return kds * glm::max(0.0f, stress_gravity - stress_viscous - stress_yield);

}

__device__ float limit(float transfer, const float mass, const float hdiff, const vec3 scale){

  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  if(transfer > 0.0f){
    if(hdiff < 0.0f){
      const float maxtransfer = 0.5f * glm::abs(hdiff) * Ac;
      if(transfer > maxtransfer)
        transfer = maxtransfer;
    } else {
      const float maxtransfer = 0.0f * glm::abs(hdiff) * Ac;
      if(transfer > maxtransfer)
      transfer = maxtransfer;
    }
  }

  else if(transfer < 0.0f){
    const float maxtransfer = hdiff * Ac;
    if(transfer < -maxtransfer)
      transfer = -maxtransfer;
  }

  transfer = glm::min(transfer, mass);
  return transfer;

}

//
// Solution Loop Functions
//

//! Initialize Particle Data from Model
__device__ void init(map_t& map, data_t& data, const param_t& param, debris_t& part, const size_t n){

  const vec3 scale = map.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float R = param.rainfall;        // Rainfall Amount  [m/y]
  const float g = param.gravity;         // Specific Gravity [m/s^2]
  const float nu = param.viscosity;      // Kinematic Viscosity [m^2/s]
  const float rho = param.debrisDensity;

  const float dt = param.timeStep;

  const vec2 normal = __normal(map, part.pos, scale);
  part.speed = g * normal;
  part.dspeed = part.speed; //!< Velocity Rate [m^2/s^2]

  const float mass = data.debris[part.ind];
  const vec2 momentum = data.debris_momentum[part.ind];
  const vec2 mspeed = __avespeed(momentum / rho, mass);
  const vec2 grad = __grad(map, part.pos, scale);
  const float slope = __hslope(map, param, part.pos, -grad);

  float suspend = debris::landslide_suspend(param, slope) * Ac;
  suspend += debris::suspend(param, mass, slope, momentum) * Ac;
  part.mass = suspend;

}

__device__ void move(const map_t& map, debris_t& part){

  const vec3 scale = map.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  if(glm::length(part.speed) == 0.0f)
    return;

  const float ds = glm::length(cl)/glm::length(part.speed);
  part.pos = part.pos + ds * (part.speed / cl);
  part.ind = __nearest(map, part.pos);

}

//! Integrate Sub-Solution Quantities in Quasi-Static Time
__device__ void integrate(const map_t& map, const param_t& param, debris_t& part){

  const vec3 scale = map.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  
  const float g = param.gravity;          // Specific Gravity [m/s^2]
  const float tau = param.debrisBedShear; // Shear-Stress Bed-Shear
  const float k2 = 0.0f;                  // Shear-Stress Viscosity

  const float ds = glm::length(cl)/glm::length(part.speed);
  
  //! Explicit Euler Forward Integration for Gravity
  const vec2 grad = __grad(map, part.pos, scale);
  part.speed = part.speed - ds * g * grad;
  
  const float shear = tau * glm::length(cl);
  part.speed = part.speed * __expf(-shear);
  part.dspeed = part.dspeed * __expf(-shear);

}

//! Mass-Transfer Characteristic Integration
__device__ void integrate_mt(const map_t& map, const param_t& param, debris_t& part){

  const vec3 scale = map.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

//  const float ds = 10.0f;//glm::length(cl);///glm::length(part.speed);
//  float deposit = debris::deposit(param, ds, part.mass, 0.0f, vec2(0.0f), scale);
//  deposit = glm::min(deposit, part.mass);
//  part.mass -= deposit;

}

//! Debris-Flow Estimate Accumulation
__device__ void track(data_t& track, const debris_t& part){

  atomicAdd(&track.debris[part.ind], (part.mass)/part.Q);
  atomicAdd(&track.debris_momentum[part.ind].x, (part.mass*part.dspeed.x)/part.Q);
  atomicAdd(&track.debris_momentum[part.ind].y, (part.mass*part.dspeed.y)/part.Q);

}

//
// Kernels
//

__global__ void solve(map_t map, data_t data, data_t track, const size_t N, const param_t param) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) 
    return;

  debris_t part;                        //!< Data along Trajectory / Per-Particle
  __sample(part, map, n, N);            //!< Sample the Trajectory
  debris::init(map, data, param, part, n); //!< Initialize Particle Properties

  // Note: Parameterize
  for(size_t age = 0; age < 256; ++age) {

    debris::track(track, part); //!< Accumulate Estimate
    debris::move(map, part);    //!< Move Trajectory
    if(__oob(map, part.pos))
      break;
    
    debris::integrate_mt(map, param, part); //!< Integrate Mass-Transfer
    debris::integrate(map, param, part);    //!< Integrate Trajectory
    if(glm::length(part.speed) == 0.0f)
      break;
  
  }

}

__global__ void mt(map_t map, data_t data, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= map.elem)
    return;

  const vec3 scale = map.scale * 1E3f;    // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]
  const float rho = param.debrisDensity;
  
  const vec2 pos = __topos(map, n);
  const float mass = data.debris[n];               // Suspended Mass Function
  const vec2 momentum = data.debris_momentum[n];
  const vec2 mspeed = __avespeed(momentum / rho, mass);
  const vec2 grad = __grad(map, pos, scale);
  const float slope = __hslope(map, param, pos + vec2(0.5f), -grad);

  const float dt = param.timeStep;
  const float landslide = dt * debris::landslide_suspend(param, slope) * Ac;
  const float deposit = debris::deposit(param, dt, mass, momentum, scale);
  const float suspend = dt * debris::suspend(param, mass, slope, momentum) * Ac;

  float transfer = (deposit - suspend - landslide);
  transfer = debris::limit(transfer, mass, (slope - param.critSlope)*glm::length(cl), scale);
  map.transfer[n] += transfer;

}

}

} // end of namespace soil

#endif