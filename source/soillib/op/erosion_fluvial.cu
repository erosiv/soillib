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

__device__ vec2 __avespeed(const model_t& model, const particle_t& part){
  
  const float discharge = model.discharge[part.ind];
  if(discharge < 1.0f)
    return vec2(0.0f);
  
  const vec2 momentum = model.momentum[part.ind];
  return momentum / discharge;

}

//
// Core Procedures
// 

//! Initialize Particle Data from Model
__device__ void init(model_t& model, const param_t& param, particle_t& part){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float& R = param.rainfall;        // Rainfall Amount  [m/y]
  const float& g = param.gravity;         // Specific Gravity [m/s^2]
  const float& nu = param.viscosity;      // Kinematic Viscosity [m^2/s]

  // Initial Velocity Estimate
  const vec3 normal = __normal(model, part.pos, scale);
  const vec2 average_speed = __avespeed(model, part);
  part.speed = g * vec2(normal.x, normal.y) + nu * average_speed;

  // Initial Tracking Values

  part.vol = Ac * R;        //!< Volume Rate [m^3/s]
  part.dspeed = part.speed; //!< Velocity Rate [m^2/s^2]

  // Initial Sediment Value:
  // Note that there is a maximum amount that can theoretically
  //  be suspended, which is when it is in balance with the amount
  //  that would also be deposited. We can use this to cap the value.
  const float discharge = model.discharge[part.ind];                // Discharge Function
  const float slope = __slope(model, param, part.pos, part.speed);  // Local Slope Function
  const float suspend = fluvial::suspend(param, part.speed*discharge, discharge, slope, part.vol, Ac);
  part.sed = suspend;

}

//! Move the Particle along the Trajectory
__device__ void move(const model_t& model, particle_t& part){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  const float ds = glm::length(cl)/glm::length(part.speed);

  part.pos = part.pos + ds * (part.speed / cl);
  part.ind = __nearest(model, part.pos);

}

//! Integrate Sub-Solution Quantities in Quasi-Static Time
__device__ void integrate(const model_t& model, const param_t& param, particle_t& part){
  
  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  
  const float g = param.gravity;          // Specific Gravity [m/s^2]
  const float k1 = param.bedShear;        // Shear-Stress Bed-Shear
  const float k2 = param.viscosity;       // Shear-Stress Viscosity
  
  // Dynamic Time-Step [s]
  const float ds = glm::length(cl)/glm::length(part.speed);
  
  const vec3 normal = __normal(model, part.pos, scale);
  const vec2 average_speed = __avespeed(model, part);
  
  //! Explicit Euler Forward Integration for Gravity
  part.speed = part.speed + ds * g * vec2(normal.x, normal.y);

  //! Implicit Euiler Forward Integration for Bed Shear-Stress and Viscosity
  part.speed =  1.0f/(1.0f + ds * (k1+k2))*part.speed + ds*k2/(1.0f + ds*(k1+k2))*average_speed;
  part.dspeed = 1.0f/(1.0f + ds * (k1+k2))*part.dspeed;

  //! Implicit Euler Forward Integration for Volume Evaporation
  part.vol = 1.0f/(1.0f + ds*param.evapRate)*part.vol;

}

//! Fluvial Erosion Mass-Transfer System
__device__ void integrate_mt(model_t& model, const param_t& param, particle_t& part){

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
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
__device__ void track(model_t& model, const particle_t& part){

  atomicAdd(&model.mass_track[part.ind], (part.sed)/part.Q);
  atomicAdd(&model.discharge_track[part.ind], (part.vol)/part.Q);
  atomicAdd(&model.momentum_track[part.ind].x, (part.vol*part.dspeed.x)/part.Q);
  atomicAdd(&model.momentum_track[part.ind].y, (part.vol*part.dspeed.y)/part.Q);

}

//
// Kernels
//

//! Transport Estimate Solution Kernel
__global__ void solve(model_t model, const size_t N, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) 
    return;

  particle_t part;                    //!< Data along Trajectory / Per-Particle
  __sample(part, model, n, N);        //!< Sample the Trajectory
  fluvial::init(model, param, part);  //!< Initialze Differential Quantities

  // Iteratively Integrate along Trajectory
  for(int age = 0; age < param.maxage; ++age){

    fluvial::track(model, part);  //!< Accumulate Estimate
    fluvial::move(model, part);   //!< Move Trajectory
    if(model.index.oob(part.pos))
      break;

    fluvial::integrate_mt(model, param, part); //!< Integrate Mass-Transfer
    fluvial::integrate(model, param, part);    //!< Integrate Differential Equation
    if(glm::length(part.speed) == 0.0f)
      break;

  }

}

//! Mass-Transfer Application Kernel
__global__ void mt(model_t model, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= model.height.elem())
    return;

  const vec3 scale = model.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  const float mass = model.mass[n];                 // Suspended Mass Function
  const float discharge = model.discharge[n];       // Discharge Function
  const vec2 momentum = model.momentum[n];
  const vec2 pos = model.index.unflatten(n);
  const float slope = __slope(model, param, pos + vec2(0.5), momentum); // Local Slope Function

//  const float dt = param.timeStep;        // Geological Timestep [y]
  const float deposit = fluvial::deposit(param, mass, discharge);
  const float suspend = fluvial::suspend(param, momentum, discharge, slope, discharge, Ac);
  float transfer = (deposit - suspend);
  transfer = fluvial::limit(transfer, mass, slope, scale);
  __transfer(model, n, transfer, Z);
  
}


} // end of namespace fluvial
} // end of namespace soil

#endif