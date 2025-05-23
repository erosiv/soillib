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

//! Suspension Rate [m^3/s]
__device__ float landslide_suspend(const param_t& param, const float hdiff, const vec3 scale) {

  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float g = param.gravity;            //!< Specific Gravity [m/s^2]
  const float kdl = param.debrisCreepRate;  //!< Landslide Erosion Rate [1/s]
  const float rho = param.debrisDensity;    //!< Debris Density [kg / m^3]

  const float slopediff = hdiff / glm::length(cl);
  const float landslide = kdl * g * glm::max(0.0f, slopediff) * Ac ;
  return landslide;

}

//! Deposition Rate [m^3/s]
__device__ float deposit(const param_t& param, const float dt, const float mass, const float hdiff, const vec2 momentum, const vec3 scale) {

  const float kdd = param.debrisDepositionRate; //!< Thermal Deposition Rate [1/s]
  const float decay = (1.0f-__expf(-dt * kdd));      //!< Total Decay Factor []
  return decay * mass;

}

//! Suspension Rate [m^3/s]
__device__ float suspend(const param_t& param, const float mass, const float hdiff, const vec2 momentum, const vec3 scale) {

  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float kds = param.debrisSuspensionRate; //!< Thermal Deposition Rate [1/s]
  const float rho = param.debrisDensity;        //!< Debris Density
  const float g = param.gravity;                //!< Specific Gravity [m/s^2]
  const float tau = param.debrisYieldStress;    //!< Debris-Flow Bed Shear Pa s
  const float mu = param.debrisViscosity; //!< Debris Flow Viscosity

  const float slopediff = hdiff / glm::length(cl);
  const float sdiff = g * mass * slopediff;
  const float stress_yield = tau / rho;
//  const float stress_viscous =  mu * glm::length(speed) / height;
  return kds * glm::max(0.0f, sdiff - stress_yield) * Ac;

//  const float height = (glm::length(momentum) / glm::length(speed)) / Ac;
//  const float abrade = rho * g * height * hdiff;
//  const float stable = rho * g * height * param.critSlope;
//  const float shear = tau + stable + mu * glm::length(speed) / height;
//  
//  return kds * glm::max(0.0f, shear - abrade) / rho;// / glm::length(speed);
  
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

  if(transfer < 0.0f){
    const float maxtransfer = glm::max(0.0f, hdiff) * Ac;
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
__device__ void init(map_t& map, data_t& data, const param_t& param, debris_t& part){

  const vec3 scale = map.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float& R = param.rainfall;        // Rainfall Amount  [m/y]
  const float& g = param.gravity;         // Specific Gravity [m/s^2]
  const float& nu = param.viscosity;      // Kinematic Viscosity [m^2/s]

  const float dt = param.timeStep;

  const vec2 normal = __normal(map, part.pos, scale);
  part.speed = g * normal;
  part.dspeed = part.speed; //!< Velocity Rate [m^2/s^2]

  const float mass = data.debris[part.ind];
  const vec2 momentum = data.debris_momentum[part.ind];

  float hdiff = __hdiff(map, param, part.pos);
  float suspend = debris::landslide_suspend(param, hdiff, scale);
  suspend += debris::suspend(param, mass, hdiff, momentum, scale);
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
  const float k1 = param.debrisBedShear;  // Shear-Stress Bed-Shear
  const float k2 = 0.0f;                  // Shear-Stress Viscosity

  const float ds = glm::length(cl)/glm::length(part.speed);
  
  //! Explicit Euler Forward Integration for Gravity
  const vec2 grad = __grad(map, part.pos, scale);
  part.speed = part.speed - ds * g * grad;
  
  const float shear = k1 * glm::length(cl);
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
  debris::init(map, data, param, part); //!< Initialize Particle Properties

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

  const float mass = data.debris[n];               // Suspended Mass Function
  const vec2 pos = __topos(map, n);
  const float hdiff = __hdiff(map, param, pos + vec2(0.5f));
  const vec2 momentum = data.debris_momentum[n];

  const float dt = param.timeStep;
  const float landslide = dt * debris::landslide_suspend(param, hdiff, scale);
  const float deposit = debris::deposit(param, dt, mass, hdiff, momentum, scale);
  const float suspend = dt * debris::suspend(param, mass, hdiff, momentum, scale);

  float transfer = (deposit - suspend - landslide);
  transfer = debris::limit(transfer, mass, hdiff, scale);
  map.transfer[n] += transfer;

}

}

} // end of namespace soil

#endif