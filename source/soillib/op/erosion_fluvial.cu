#ifndef SOILLIB_OP_EROSION_FLUVIAL_CU
#define SOILLIB_OP_EROSION_FLUVIAL_CU
#define HAS_CUDA

#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <soillib/op/erosion.hpp>
#include <soillib/op/gather.hpp>

#include <soillib/op/erosion_map.cu>

namespace soil {

//
// Local Particle State
//

struct particle_t {

  vec2 pos;   //!< World Position [pix]
  vec2 speed; //!< World Velocity [m/s]

  float Q;  //!< Weighted Sampling Probability
  int ind;  //!< Nearest Support Index

  vec2 dspeed;  //!< Characteristic Speed Rate
  float vol;    //!< Characteristic Volume Rate
  float sed;    //!< Characteristic Mass Rate

};

namespace fluvial {

//
// Mass-Transfer Functions
//
//  Note that in the transport rates, the value is not clamped, since
//  value-clamping and limiting only occurs in the composite sum.
//

//! Mass Deposition Rate
__device__ float deposit(const param_t& param, const float mass, const float discharge){

  if(discharge < 1.0f)
    return 0.0f;

  const float kd = param.depositionRate;        // Fluvial Deposition Rate [1/y]
  const float deposit = kd * mass;// / discharge;  // Deposited Mass [kg]
  return deposit;

}

//! Mass Suspension Rate
__device__ float suspend(const param_t& param, const vec2 momentum, const float discharge, const float slope, const float vol, const float Area){

  if(discharge < 1.0f)
    return 0.0f;
  
  const float alpha = 0.1333f;
  const float fD = 0.1f;                  //!< Darcy-Weisbach Friction Factor
  const float rho = 1.0f;                 //!< Density of Fluid [kg/m^3]
  const float ks = param.suspensionRate;  // Fluvial Suspension Rate [(m^3/y)^-0.4]
  
  const float velocity = glm::length(momentum / discharge);
  const float shear = 0.125f * fD * rho * velocity * velocity;
  const float power = pow(shear * velocity, alpha); // Stream Power Function
  const float suspend = ks * power * vol;           // Concentration
  
  const float mask = (slope < 0.0f)?1.0f:0.0f;  // Activation Function
  return mask * suspend;                        // [kg/s] (Activated)

}

// Note: Maxtransfer here is damped for stability. This should be
//  attempted to be removed using alternative stabilizing methods.
__device__ float limit(float transfer, const float mass, const float slope, const vec3 scale){

  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  if(transfer <= 0.0f){
    const float maxtransfer = 0.1f * slope * glm::length(cl) / scale.z * Z;
    const float tmin = transfer * glm::min(1.0f, glm::abs(maxtransfer/transfer));
    transfer = glm::max(transfer, tmin);
  }

  transfer = glm::min(transfer, mass);  // Limit by Mass
  return transfer;

}

__device__ vec2 __avespeed(const vec2 momentum, const float discharge){
  
  if(discharge < 1.0f)
    return vec2(0.0f);
  return momentum / discharge;

}

//
// Core Procedures
// 

//! Initialize Particle Data from Model
__device__ void init(map_t& map, data_t& data, const param_t& param, particle_t& part){

  const vec3 scale = map.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float& R = param.rainfall;        // Rainfall Amount  [m/y]
  const float& g = param.gravity;         // Specific Gravity [m/s^2]
  const float& nu = param.viscosity;      // Kinematic Viscosity [m^2/s]

  // Initial Velocity Estimate
  const float discharge = data.discharge[part.ind];
  const vec2 momentum = data.momentum[part.ind];
  const vec2 average_speed = __avespeed(momentum, discharge);
  const vec3 normal = __normal(map, part.pos, scale);
  part.speed = g * vec2(normal.x, normal.y) + nu * average_speed;

  // Initial Tracking Values

  part.vol = Ac * R;        //!< Volume Rate [m^3/s]
  part.dspeed = part.speed; //!< Velocity Rate [m^2/s^2]

  // Initial Sediment Value:
  // Note that there is a maximum amount that can theoretically
  //  be suspended, which is when it is in balance with the amount
  //  that would also be deposited. We can use this to cap the value.
  const float slope = __slope(map, param, part.pos, part.speed);  // Local Slope Function
  const float suspend = fluvial::suspend(param, part.speed*discharge, discharge, slope, part.vol, Ac);
  part.sed = suspend;

}

//! Move the Particle along the Trajectory
__device__ void move(const map_t& map, const data_t& data, particle_t& part){

  const vec3 scale = map.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  const float ds = glm::length(cl)/glm::length(part.speed);

  part.pos = part.pos + ds * (part.speed / cl);
  part.ind = __nearest(map, part.pos);

}

//! Integrate Sub-Solution Quantities in Quasi-Static Time
__device__ void integrate(const map_t& map, const data_t& data, const param_t& param, particle_t& part){
  
  const vec3 scale = map.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  
  const float g = param.gravity;          // Specific Gravity [m/s^2]
  const float k1 = param.bedShear;        // Shear-Stress Bed-Shear
  const float k2 = param.viscosity;       // Shear-Stress Viscosity
  
  // Dynamic Time-Step [s]
  const float ds = glm::length(cl)/glm::length(part.speed);
  
  const float discharge = data.discharge[part.ind];
  const vec2 momentum = data.momentum[part.ind];
  const vec2 average_speed = __avespeed(momentum, discharge);
  const vec3 normal = __normal(map, part.pos, scale);
  
  //! Explicit Euler Forward Integration for Gravity
  part.speed = part.speed + ds * g * vec2(normal.x, normal.y);

  //! Implicit Euiler Forward Integration for Bed Shear-Stress and Viscosity
  part.speed =  1.0f/(1.0f + ds * (k1+k2))*part.speed + ds*k2/(1.0f + ds*(k1+k2))*average_speed;
  part.dspeed = 1.0f/(1.0f + ds * (k1+k2))*part.dspeed;

  //! Implicit Euler Forward Integration for Volume Evaporation
  part.vol = 1.0f/(1.0f + ds*param.evapRate)*part.vol;

}

//! Fluvial Erosion Mass-Transfer System
__device__ void integrate_mt(map_t& map, data_t& data, const param_t& param, particle_t& part){

  const vec3 scale = map.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  const float ds = glm::length(cl)/glm::length(part.speed); // Dynamic Time-Step
  
  const float mass = part.sed;
  float deposit = param.depositionRate * mass;// / part.vol;
  deposit = glm::min(deposit, mass);
  part.sed -= deposit;

}

//! Track the Differential Quantities along Trajectories
__device__ void track(data_t& data, const particle_t& part){

  atomicAdd(&data.mass_track[part.ind], (part.sed)/part.Q);
  atomicAdd(&data.discharge_track[part.ind], (part.vol)/part.Q);
  atomicAdd(&data.momentum_track[part.ind].x, (part.vol*part.dspeed.x)/part.Q);
  atomicAdd(&data.momentum_track[part.ind].y, (part.vol*part.dspeed.y)/part.Q);

}

//
// Kernels
//

//! Transport Estimate Solution Kernel
__global__ void solve(map_t map, data_t data, const size_t N, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) 
    return;

  particle_t part;                    //!< Data along Trajectory / Per-Particle
  __sample(part, map, &data.rand[n], n, N);        //!< Sample the Trajectory
  fluvial::init(map, data, param, part);  //!< Initialze Differential Quantities

  // Iteratively Integrate along Trajectory
  for(int age = 0; age < param.maxage; ++age){

    fluvial::track(data, part);  //!< Accumulate Estimate
    fluvial::move(map, data, part);   //!< Move Trajectory
    if(map.index.oob(part.pos))
      break;

    fluvial::integrate_mt(map, data, param, part); //!< Integrate Mass-Transfer
    fluvial::integrate(map, data, param, part);    //!< Integrate Differential Equation
    if(glm::length(part.speed) == 0.0f)
      break;

  }

}

//! Mass-Transfer Application Kernel
__global__ void mt(map_t map, data_t data, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= map.height.elem())
    return;

  const vec3 scale = map.scale * 1E3f;    // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  const float mass = data.mass[n];                 // Suspended Mass Function
  const float discharge = data.discharge[n];       // Discharge Function
  const vec2 momentum = data.momentum[n];
  const vec2 pos = map.index.unflatten(n);
  const float slope = __slope(map, param, pos + vec2(0.5), momentum); // Local Slope Function

//  const float dt = param.timeStep;        // Geological Timestep [y]
  const float deposit = fluvial::deposit(param, mass, discharge);
  const float suspend = fluvial::suspend(param, momentum, discharge, slope, discharge, Ac);
  float transfer = (deposit - suspend);
  transfer = fluvial::limit(transfer, mass, slope, scale);
  __transfer(map, n, transfer, Z);
  
}

} // end of namespace fluvial
} // end of namespace soil

#endif