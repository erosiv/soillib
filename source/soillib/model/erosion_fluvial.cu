#ifndef SOILLIB_MODEL_EROSION_FLUVIAL_CU
#define SOILLIB_MODEL_EROSION_FLUVIAL_CU
#define HAS_CUDA

#include <silt/core/types.hpp>
#include <silt/core/tensor.hpp>
#include <silt/op/gather.hpp>

#include <soillib/model/erosion.hpp>
#include <soillib/model/erosion_map.cu>

// Soillib Fluvial Erosion Implementation
//
// ... add description ...

namespace soil {

//! Fluvial Particle State
struct particle_t {

  vec2 pos;     //!< Grid-Space Position [pix]
  int ind;      //!< Nearest Support Index
  float Q;      //!< Weighted Sampling Probability
  
  vec2 speed;   //!< World-Space Velocity [m/s]
  vec2 dspeed;  //!< Speed Rate  [m/s^2]
  float vol;    //!< Volume Rate [m^3/s]
  float sed;    //!< Mass Rate   [kg/s]

};

namespace fluvial {
using namespace silt;

//
// Mass-Transfer Functions
//
//  Note that in the transport rates, the value is not clamped, since
//  value-clamping and limiting only occurs in the composite sum.
//

//! Mass Deposition Rate [m/y]
__device__ float deposit(const param_t& param, const float dt, const float mass, const float discharge){

  if(discharge < 1.0f)
    return 0.0f;

  const float kd = param.depositionRate;  //!< Fluvial Deposition Rate [m/y]
  return dt * kd * mass / discharge;

}

//! Mass Suspension Rate [m/y]
__device__ float suspend(const param_t& param, const vec2 speed, const float vol){

  const float alpha = param.fluvialExponent;
  const float fD = param.frictionFactor;      //!< Darcy-Weisbach Friction Factor
  const float rho = param.fluvialDensity;     //!< Density of Fluid [kg/m^3]
  const float ks = param.suspensionRate;      //!< Fluvial Suspension Rate [(m^3/y)^-0.4]
  
  const float velocity = glm::length(speed);                    //!< [m/s]
  const float shear = 0.125f * fD * rho * velocity * velocity;  //!< [kg/m/s^2]
  const float power = pow(shear * velocity, alpha);             //!< Stream Power Function
  const float suspend = ks * power;
  return glm::abs(suspend);

}

// Note: Maxtransfer here is damped for stability. This should be
//  attempted to be removed using alternative stabilizing methods.
__device__ float limit(float transfer, const float mass, const float slope, const scale_t scale){

  if(transfer <= 0.0f){
    const float maxtransfer = 0.1f * slope * scale.len * scale.Ac;
    if(transfer < maxtransfer){
      transfer = maxtransfer;
    }
  }

//  transfer = glm::min(transfer, mass);  // Limit by Mass
  return transfer;

}

//
// Core Procedures
// 

//! Initialize Particle Data from Model
//!
//! Note that there is a maximum amount that can theoretically
//!  be suspended, which is when it is in balance with the amount
//!  that would also be deposit1ed. We can use this to cap the value.
__device__ void init(
  particle_t& part,
  const map_t& map,
  const data_t& data,
  const param_t& param,
  const scale_t& scale
){

  const float g = param.gravity;                            //!< Specific Gravity [m/s^2]
  const float nu = param.viscosity;                         //!< Kinematic Viscosity [m^2/s]
  const float rho = param.fluvialDensity;                   //!< Fluvial Density [kg/m^3]
  const float R = param.rainfall * map.rainfall[part.ind];  //!< Rainfall Rate  [m/y]

  // Initial Velocity Estimate
  const float discharge = data.discharge[part.ind];
  const auto view =  data.momentum.view<vec2>();
  const vec2 momentum = view[part.ind];
  const vec2 mspeed = __avespeed(momentum / rho, discharge);
  const vec2 grad = __grad(map, part.pos, scale); //!< Scaled Direction
  const float suspend = fluvial::suspend(param, mspeed, scale.Ac * R);
  
  // Initial Tracking Values
  part.vol = scale.Ac * R;                             //!< Volume Rate [m^3/s]
  part.dspeed = nu * mspeed - g * grad + param.force;  //!< Velocity Rate [m/s^2]
  part.speed = part.dspeed;                            //!< Velocity [m/s]
  part.sed = scale.Ac * suspend;

}

//! Move the Particle along the Trajectory
__device__ void move(const map_t& map, particle_t& part, const scale_t& scale){

  const float ds = scale.len/glm::length(part.speed);
  part.pos = part.pos + ds * (part.speed / scale.cl);
  part.ind = map.shape.flatten(part.pos);

}

__device__ void integrate(
  const map_t& map,
  const data_t& data,
  const param_t& param,
  particle_t& part,
  const scale_t& scale
){

  const float g = param.gravity;          //!< Specific Gravity [m/s^2]
  const float tau = param.bedShear;       //!< Shear-Stress Bed-Shear [kg/m/s^2]
  const float nu = param.viscosity;       //!< Shear-Stress Viscosity [m^2/s]
  const float ke = param.evapRate;        //!< Water Evaporation Rate [1/s]
  const float rho = param.fluvialDensity; //!< Water Density [kg/m^3]
  const vec2 force = param.force;         //!< Body Force [m/s^2]

  const float discharge = data.discharge[part.ind];                 //!< [m^3/s]
  const vec2 momentum = data.momentum.view<vec2>()[part.ind];       //!< [kg*m/s]
  const vec2 ave_speed = __avespeed(momentum / rho, discharge); //!< [m/s]
  const vec2 grad = __grad(map, part.pos, scale);

  // Dynamic Time-Step [s]
  
  float ds = scale.len;///glm::length(part.speed);
  part.vol *= __expf(-ds * param.evapRate); //!< Evaporate Volume
  
  // Velocity Update
  
  const float mu_t = ds * nu * glm::length(momentum/rho); //!< Momentum Transfer
  const float mu_w = 1.0f / (1.0f + mu_t);                //!< Implicit Momentum Mix Factor
  part.speed = mu_w * part.speed + (1.0f-mu_w) * ave_speed;

  part.speed = part.speed - ds * g * grad;
  part.speed = part.speed + ds * force;
  
  const float tau_decay = ds * tau / rho;
  part.speed = part.speed * __expf( -tau_decay );
  part.dspeed = part.dspeed * __expf( -tau_decay );

}

//! Fluvial Erosion Mass-Transfer System:
//!   Exponentially Decay the Mass-Rate Value
__device__ void integrate_mt(map_t& map, data_t& data, const param_t& param, particle_t& part){

//  const vec3 scale = map.scale * 1E3f;    //!< Cell Scale [m] (conv. from km)
//  const vec2 cl = vec2(scale.x, scale.y); //!< Cell Length [m, m]

//  const float kd = 0.05f;//param.depositionRate;                //!< Fluvial Deposition Rate [1/s]
//  const float ds = glm::length(cl);///glm::length(part.speed);  //!< Dynamic Timestep [s]
//  part.sed *= __expf(-kd);                                      //!< Apply Decay

}

//! Track the Differential Quantities along Trajectories
__device__ void track(data_t& track, const particle_t& part) {

  const float rho = 1000.0f;    //!< Fluid Density
  const float rho_s = 2500.0f;  //!< Sediment Density

  atomicAdd(&track.mass[part.ind], (part.sed)/part.Q);
  atomicAdd(&track.discharge[part.ind], (part.vol)/part.Q);

  auto view = track.momentum.view<vec2>();
  atomicAdd(&view[part.ind].x, (part.vol*rho*part.dspeed.x)/part.Q);
  atomicAdd(&view[part.ind].y, (part.vol*rho*part.dspeed.y)/part.Q);

}

//
// Kernels
//

//! Transport Solution Kernel
__global__ void solve(
  map_t map,
  data_t data,
  data_t track,
  const param_t param,
  const scale_t scale,
  const size_t n_samples
){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= n_samples)
    return;

  // Initialize the Particle
  particle_t part;                              //!< Data along Trajectory / Per-Particle
  __sample(part, map, n, n_samples);            //!< Sample the Trajectory
  fluvial::init(part, map, data, param, scale); //!< Initialze Differential Quantities

  // Iteratively Integrate along Trajectory
  for(int age = 0; age < param.maxage; ++age){

    fluvial::track(track, part);      //!< Accumulate Estimate
    fluvial::move(map, part, scale);  //!< Move Trajectory
    if(__oob(map, part.pos))
      break;

//    fluvial::integrate_mt(map, data, param, part);    //!< Integrate Mass-Transfer
    fluvial::integrate(map, data, param, part, scale);  //!< Integrate Differential Equation
    if(glm::length(part.speed) < 1E-6*scale.len)
      break;

  }

}

//! Mass-Transfer Application Kernel
__global__ void mt(map_t map, data_t data, const param_t param, const scale_t scale){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= map.elem)
    return;

  const float dt = param.timeStep;        // Geological Timestep [y]
  const float rho = param.fluvialDensity;

  const float mass = data.mass[n];                 // Suspended Mass Function
  const float discharge = data.discharge[n];       // Discharge Function
  const auto view =  data.momentum.view<vec2>();
  const vec2 momentum = view[n];

  const vec2 pos = __topos(map, n);
  const vec2 mspeed = __avespeed(momentum / rho, discharge);
  const float deposit = fluvial::deposit(param, dt * scale.len, mass, discharge);
  const float suspend = dt * fluvial::suspend(param, mspeed, discharge) * scale.Ac;
  const float slope = __slope(map, param, pos + vec2(0.5), momentum, scale); // Local Slope Function

  float transfer = (deposit - suspend);
  transfer = fluvial::limit(transfer, mass, slope, scale);
  map.transfer[n] += transfer;

}

} // end of namespace fluvial
} // end of namespace soil

#endif