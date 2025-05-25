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

  vec2 pos;     //!< Grid-Space Position [pix]
  vec2 speed;   //!< World-Space Velocity [m/s]

  vec2 dspeed;  //!< Characteristic Speed Rate  [m/s^2]
  float vol;    //!< Characteristic Volume Rate [m^3/s]
  float sed;    //!< Characteristic Mass Rate   [kg/s]

  int ind;      //!< Nearest Support Index
  float Q;      //!< Weighted Sampling Probability

};

namespace fluvial {

//
// Mass-Transfer Functions
//
//  Note that in the transport rates, the value is not clamped, since
//  value-clamping and limiting only occurs in the composite sum.
//

//! Mass Deposition Rate:
__device__ float deposit(const param_t& param, const float dt, const float mass, const float discharge){

  if(discharge < 1.0f)
    return 0.0f;

  const float kd = param.depositionRate;      //!< Fluvial Deposition Rate [1/y]
  const float decay = (1.0f-__expf(-dt*kd));  //!< Total Decay Factor []
  return decay * mass;

}

//! Mass Suspension Rate [m/y]
__device__ float suspend(const param_t& param, const vec2 speed, const float vol){

  const float alpha = param.fluvialExponent; 
  const float fD = param.frictionFactor;      //!< Darcy-Weisbach Friction Factor
  const float rho = param.fluvialDensity;     //!< Density of Fluid [kg/m^3]
  const float ks = param.suspensionRate;      //!< Fluvial Suspension Rate [(m^3/y)^-0.4]
  
  const float velocity = glm::length(speed);  //!< [m/s]
  const float shear = 0.125f * fD * rho * velocity * velocity;      //!< [kg/m/s^2]
  const float power = pow(shear * velocity, alpha);                 //!< Stream Power Function
  const float suspend = ks * power * vol;                           //!< Concentration
  return glm::abs(suspend);

}

// Note: Maxtransfer here is damped for stability. This should be
//  attempted to be removed using alternative stabilizing methods.
__device__ float limit(float transfer, const float mass, const float slope, const vec3 scale){

  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  if(transfer <= 0.0f){
    const float maxtransfer = 0.1f * slope * glm::length(cl) * Ac;
    if(transfer < maxtransfer){
      transfer = maxtransfer;
    }
  }

  transfer = glm::min(transfer, mass);  // Limit by Mass
  return transfer;

}

//
// Core Procedures
// 

//! Initialize Particle Data from Model
template<typename Map>
__device__ void init(Map& map, data_t& data, const param_t& param, particle_t& part){

  const vec3 scale = map.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]

  const float g = param.gravity;          //!< Specific Gravity [m/s^2]
  const float nu = param.viscosity;       //!< Kinematic Viscosity [m^2/s]
  const float rho = param.fluvialDensity; //!< Fluvial Density [kg/m^3]

  // Initial Velocity Estimate
  const float discharge = data.discharge[part.ind];
  const vec2 momentum = data.momentum[part.ind];
  const vec2 mspeed = __avespeed(momentum / rho, discharge);
  const vec2 grad = __grad(map, part.pos, scale); //!< Scaled Direction
  
  // Initial Tracking Values
  
  const float& R = param.rainfall;            //!< Rainfall Rate  [m/y]
  const float Rmask = map.rainfall[part.ind]; //!< Rainfall Mask
  part.vol = Ac * R * Rmask;                  //!< Volume Rate [m^3/s]

  const vec2 force = param.force;
  part.dspeed = nu * mspeed - g * grad + param.force;  //!< Velocity Rate [m/s^2]
  part.speed = part.dspeed;                                   //!< Velocity [m/s]

  // Initial Sediment Value:
  // Note that there is a maximum amount that can theoretically
  //  be suspended, which is when it is in balance with the amount
  //  that would also be deposit1ed. We can use this to cap the value.
  const float suspend = fluvial::suspend(param, mspeed, part.vol);
  part.sed = suspend * Ac;

}

//! Move the Particle along the Trajectory
__device__ void move(const map_grid& map, particle_t& part){

  const vec3 scale = map.scale * 1E3f;    // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  const float ds = glm::length(cl)/glm::length(part.speed);

  part.pos = part.pos + ds * (part.speed / cl);
  part.ind = __nearest(map, part.pos);

}

//! Integrate Sub-Solution Quantities in Quasi-Static Time
template<typename Map>
__device__ void integrate(const Map& map, const data_t& data, const param_t& param, particle_t& part){
  
  const vec3 scale = map.scale * 1E3f;    //!< Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); //!< Cell Length [m, m]
  const float Ac = scale.x*scale.y;       //!< Cell Area [m^2]

  const float g = param.gravity;          //!< Specific Gravity [m/s^2]
  const float tau = param.bedShear;       //!< Shear-Stress Bed-Shear [kg/m/s^2]
  const float nu = param.viscosity;       //!< Shear-Stress Viscosity [m^2/s]
  const float ke = param.evapRate;        //!< Water Evaporation Rate [1/s]
  const float rho = param.fluvialDensity; //!< Water Density [kg/m^3]
  const vec2 force = param.force;         //!< Body Force [m/s^2]

  const float discharge = data.discharge[part.ind];           //!< [m^3/s]
  const vec2 momentum = data.momentum[part.ind];              //!< [kg*m/s]
  const vec2 average_speed = __avespeed(momentum / rho, discharge); //!< [m/s]
  const vec2 grad = __grad(map, part.pos, scale);

  // Dynamic Time-Step [s]
  
  float ds = glm::length(cl);///glm::length(part.speed);
  part.vol *= __expf(-ds * param.evapRate); //!< Evaporate Volume
  
  // Velocity Update
  
  const float mu_t = ds * nu * glm::length(momentum/rho); //!< Momentum Transfer
  const float mu_w = 1.0f / (1.0f + mu_t);                //!< Implicit Momentum Mix Factor
  part.speed = mu_w * part.speed + (1.0f - mu_w) * average_speed;

  part.speed = part.speed - ds * g * grad;
  part.speed = part.speed + ds * force;
  
  const float tau_decay = ds * tau / rho;
  part.speed = part.speed * __expf( -tau_decay );
  part.dspeed = part.dspeed * __expf( -tau_decay );

}

//! Fluvial Erosion Mass-Transfer System:
//!   Exponentially Decay the Mass-Rate Value
template<typename Map>
__device__ void integrate_mt(Map& map, data_t& data, const param_t& param, particle_t& part){

//  const vec3 scale = map.scale * 1E3f;    //!< Cell Scale [m] (conv. from km)
//  const vec2 cl = vec2(scale.x, scale.y); //!< Cell Length [m, m]
//
//  const float kd = param.depositionRate;                    //!< Fluvial Deposition Rate [1/s]
//  const float ds = glm::length(cl)/glm::length(part.speed); //!< Dynamic Timestep [s]
//  part.sed *= __expf(-ds * param.depositionRate);           //!< Apply Decay

}

//! Track the Differential Quantities along Trajectories
__device__ void track(data_t& track, const particle_t& part) {

  const float rho = 1000.0f;  //!< Fluid Density
  atomicAdd(&track.mass[part.ind], (part.sed)/part.Q);
  atomicAdd(&track.discharge[part.ind], (part.vol)/part.Q);
  atomicAdd(&track.momentum[part.ind].x, (part.vol*rho*part.dspeed.x)/part.Q);
  atomicAdd(&track.momentum[part.ind].y, (part.vol*rho*part.dspeed.y)/part.Q);

}

//
// Kernels
//

//! Transport Estimate Solution Kernel
template<typename Map>
__global__ void solve(Map map, data_t data, data_t track, const size_t N, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= N) 
    return;

  const vec3 scale = map.scale * 1E3f;    // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  particle_t part;                        //!< Data along Trajectory / Per-Particle
  __sample(part, map, n, N);              //!< Sample the Trajectory
  fluvial::init(map, data, param, part);  //!< Initialze Differential Quantities

  // Iteratively Integrate along Trajectory
  for(int age = 0; age < param.maxage; ++age){

    fluvial::track(track, part);  //!< Accumulate Estimate
    fluvial::move(map, part);     //!< Move Trajectory
    if(__oob(map, part.pos))
      break;

    fluvial::integrate_mt(map, data, param, part); //!< Integrate Mass-Transfer
    fluvial::integrate(map, data, param, part);    //!< Integrate Differential Equation
    if(glm::length(part.speed) < 1E-6*glm::length(cl))
      break;

  }

}

//! Mass-Transfer Application Kernel
template<typename Map>
__global__ void mt(Map map, data_t data, const param_t param){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= map.elem)
    return;

  const vec3 scale = map.scale * 1E3f;    // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]
  const float dt = param.timeStep;        // Geological Timestep [y]
  const float rho = param.fluvialDensity;

  const float mass = data.mass[n];                 // Suspended Mass Function
  const float discharge = data.discharge[n];       // Discharge Function
  const vec2 momentum = data.momentum[n];
  const vec2 pos = __topos(map, n);
  const vec2 mspeed = __avespeed(momentum / rho, discharge);
  
  const float deposit = fluvial::deposit(param, dt * glm::length(cl), mass, discharge);
  const float suspend = dt * fluvial::suspend(param, mspeed, discharge) * Ac;
  
//  const vec2 grad = __grad(map, pos, scale);
//  const float slope = glm::dot(grad, glm::normalize(mspeed));
  const float slope = __slope(map, param, pos + vec2(0.5), momentum); // Local Slope Function

  float transfer = (deposit - suspend);
  transfer = fluvial::limit(transfer, mass, slope, scale);
  map.transfer[n] += transfer;

}

} // end of namespace fluvial
} // end of namespace soil

#endif