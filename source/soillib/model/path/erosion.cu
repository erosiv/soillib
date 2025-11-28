#ifndef SOILLIB_MODEL_EROSION_CU
#define SOILLIB_MODEL_EROSION_CU
#define HAS_CUDA

#include <soillib/soillib.hpp>

#include <silt/core/error.hpp>
#include <silt/core/tensor.hpp>
#include <silt/op/gather.hpp>
#include <silt/op/common.hpp>

#include <soillib/model/path/erosion.hpp>
// #include <soillib/model/path/erosion_map.cu>

#include <math_constants.h>

__device__ silt::vec2 __grad(
  const silt::tensor_t<float>& height,
  const silt::shape shape,
  const silt::vec3 scale,
  const silt::vec2 pos
){

  const int i00 = shape.flatten(pos + silt::vec2( 0, 0));
  const int in0 = shape.flatten(pos + silt::vec2(-1, 0));
  const int ip0 = shape.flatten(pos + silt::vec2( 1, 0));
  const int i0n = shape.flatten(pos + silt::vec2( 0,-1));
  const int i0p = shape.flatten(pos + silt::vec2( 0, 1));

  const float h = height[i00] * scale.z;
  const float hn0 = shape.oob(pos + silt::vec2(-1, 0)) ? h : height[in0] * scale.z;
  const float hp0 = shape.oob(pos + silt::vec2( 1, 0)) ? h : height[ip0] * scale.z;
  const float h0n = shape.oob(pos + silt::vec2( 0,-1)) ? h : height[i0n] * scale.z;
  const float h0p = shape.oob(pos + silt::vec2( 0, 1)) ? h : height[i0p] * scale.z;

  float gx = 0.0f;
  if(__isnanf(hn0)) gx = (hp0 - h)/scale.x;
  if(__isnanf(hp0)) gx = (h - hn0)/scale.x;
  if(!__isnanf(hp0) && !__isnanf(hn0)){
    if(hn0 < hp0) gx = (h - hn0)/scale.x;
    if(hp0 < hn0) gx = (hp0 - h)/scale.x;
    // if they are the same, slope is zero
  }

  float gy = 0.0f;
  if(__isnanf(h0n)) gy = (h0p - h)/scale.y;
  if(__isnanf(h0p)) gy = (h - h0n)/scale.y;
  if(!__isnanf(h0p) && !__isnanf(h0n)){
    if(h0n < h0p) gy = (h - h0n)/scale.y;
    if(h0p < h0n) gy = (h0p - h)/scale.y;
    // if they are the same, slope is zero
  }

  return silt::vec2(gx, gy);

}

__device__ float __slope (
  const silt::tensor_t<float>& height,
  const silt::shape shape,
  const silt::vec3 scale,
  const silt::vec2 pos
){

  const int i00 = shape.flatten(pos + silt::vec2( 0, 0));
  const int in0 = shape.flatten(pos + silt::vec2(-1, 0));
  const int ip0 = shape.flatten(pos + silt::vec2( 1, 0));
  const int i0n = shape.flatten(pos + silt::vec2( 0,-1));
  const int i0p = shape.flatten(pos + silt::vec2( 0, 1));

  const float h = height[i00] * scale.z;
  const float hn0 = shape.oob(pos + silt::vec2(-1, 0)) ? CUDART_NAN_F : height[in0] * scale.z;
  const float hp0 = shape.oob(pos + silt::vec2( 1, 0)) ? CUDART_NAN_F : height[ip0] * scale.z;
  const float h0n = shape.oob(pos + silt::vec2( 0,-1)) ? CUDART_NAN_F : height[i0n] * scale.z;
  const float h0p = shape.oob(pos + silt::vec2( 0, 1)) ? CUDART_NAN_F : height[i0p] * scale.z;

  // Min Gradient Computation w. Bounds Handling
  float gx = 0.0f;
  if(!__isnanf(hn0)) gx = glm::max(gx, (h - hn0)/scale.x);
  if(!__isnanf(hp0)) gx = glm::max(gx, (h - hp0)/scale.x);

  float gy = 0.0f;
  if(!__isnanf(h0n)) gy = glm::max(gy, (h - h0n)/scale.y);
  if(!__isnanf(h0p)) gy = glm::max(gy, (h - h0p)/scale.y);

  // Write to 2D vector view
  return glm::length(silt::vec2(gx, gy));

}

__global__ void __erode (
  silt::tensor_t<float> height,
  silt::tensor_t<float> discharge,
  silt::tensor_t<float> dischargeTrack,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentumTrack,
  silt::tensor_t<silt::rng> rng,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= rng.elem()) return;

  // Random Sample Position
  silt::vec2 pos = silt::vec2 {
    curand_uniform(&rng[n])*float(shape[0]),
    curand_uniform(&rng[n])*float(shape[1])
  };
  int ind = shape.flatten(pos);
  const float P = 1.0f / float(shape.elem);

  auto momentumView = momentum.view<silt::vec2>();
  auto momentumTrackView = momentumTrack.view<silt::vec2>();

  // const vecD S = sourceView[ind] / P;
  // Transport Initialization
  
  float water = 1.0f;
  float mass = 0.0f;
  silt::vec2 speed = -__grad(height, shape, scale, pos);

  // Iterate over Number of Steps
  for(int step = 0; step < param.maxage; ++step) {

    // Erosion Step / Mass Integration Step
    const float slope = __slope(height, shape, scale, pos);
    const float alpha = param.fluvialExponent;
    const float fD = param.frictionFactor;                        //!< Darcy-Weisbach Friction Factor
    const float rho = param.fluvialDensity;                       //!< Density of Fluid [kg/m^3]
    const float ks = param.suspensionRate;                        //!< Fluvial Suspension Rate [(m^3/y)^-0.4
    const float velocity = glm::length(speed);                    //!< [m/s]
    const float shear = 0.125f * fD * rho * velocity * velocity;  //!< [kg/m/s^2]
    const float power = pow(shear * velocity, alpha);             //!< Stream Power Function

    const float suspend = glm::max(0.0f, ks * power * slope);
    const float deposit = glm::min(mass, param.depositionRate * mass / water);
    const float transfer = suspend - deposit;

    atomicAdd(&height[ind], -transfer / scale.z);
    mass += transfer;
    water = (1.0f - param.evapRate) * water;

    // Position Update
    pos += speed / glm::length(speed);
    if(shape.oob(pos))
      break;

    // Tracking Step
    ind = shape.flatten(pos);
    atomicAdd(&dischargeTrack[ind], water);
    atomicAdd(&momentumTrackView[ind].x, water * speed.x);
    atomicAdd(&momentumTrackView[ind].y, water * speed.y);

    // Velocity Update
    const silt::vec2 mspeed = momentumView[ind] / discharge[ind];

    const float nu = param.viscosity;
    const float tau = param.bedShear;

    speed = (1.0f - tau) * speed;
    speed = ((1.0f - nu) * speed + nu * mspeed);
    speed = (speed - __grad(height, shape, scale, pos));
    if(glm::length(speed) < 1E-6f)
      break;

  }

}

__global__ void __erode_debris (
  silt::tensor_t<float> height,
  silt::tensor_t<float> massBuf,
  silt::tensor_t<float> massTrack,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentumTrack,
  silt::tensor_t<silt::rng> rng,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= rng.elem()) return;

  // Random Sample Position
  silt::vec2 pos = silt::vec2 {
    curand_uniform(&rng[n])*float(shape[0]),
    curand_uniform(&rng[n])*float(shape[1])
  };
  int ind = shape.flatten(pos);
  const float P = 1.0f / float(shape.elem);
  // const vecD S = sourceView[ind] / P;

//  auto momentumView = momentum.view<silt::vec2>();
//  auto momentumTrackView = momentumTrack.view<silt::vec2>();

  // Transport Initialization
  silt::vec2 speed = -__grad(height, shape, scale, pos);
  float mass = 0.0f;

  // Iterate over Number of Steps
  for(int step = 0; step < param.maxage; ++step){

    // Erosion Step
    const float slope = __slope(height, shape, scale, pos);
    const float suspend = param.debrisSuspensionRate * glm::max(0.0f, slope - param.critSlope);
    const float deposit = glm::min(mass, param.debrisDepositionRate * mass);
    const float transfer = suspend - deposit;
    atomicAdd(&height[ind], -transfer / scale.z);
    mass += transfer;

// Debris-Flow Erosion Formula
//    const float shearViscous = param.debrisViscosity * glm::length(speed) / (0.0001f + mass);
//    const float shearYield = param.debrisYieldStress + mass * param.critSlope;
//    const float shearMass = mass * (height_cur - height_next);
//    const float suspend = glm::max(0.0f, param.debrisSuspensionRate * (shearMass - shearViscous - shearYield));
//    const float deposit =  glm::min(mass, glm::max(0.0f, param.debrisDepositionRate * (shearViscous + shearYield - shearMass)));

    // Position Update
    pos += speed / glm::length(speed);
    if(shape.oob(pos))
    break;
    
    // Tracking Step
    ind = shape.flatten(pos);
    atomicAdd(&massTrack[ind], mass);
//    atomicAdd(&momentumTrackView[nind].x, mass * speed.x);
//    atomicAdd(&momentumTrackView[nind].y, mass * speed.y);

    // Velocity Update
//    const silt::vec2 mspeed = momentumView[ind] / mass[ind];
//    const float nu = param.viscosity;
    const float tau = param.bedShear;

    speed = (1.0f - tau) * speed;
//    speed = ((1.0f - nu) * speed + nu * mspeed);
    speed = (speed - __grad(height, shape, scale, pos));
    if(glm::length(speed) < 1E-6f)
      break;
  
  }

}

namespace soil {

void erode (
  silt::tensor_t<float> height,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentumTrack,
  silt::tensor_t<float> discharge,
  silt::tensor_t<float> dischargeTrack,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param
) {

  // velocity field?
  const float A = scale.x * scale.y;

  silt::set(dischargeTrack, A);
  silt::set(momentumTrack, 0.0f);

  __erode<<<block(rng.elem(), 512), 512>>>(
    height,
    discharge,
    dischargeTrack,
    momentum,
    momentumTrack,
    rng, height.shape(), scale, param);

  silt::mix(discharge, dischargeTrack, param.lrate);
  silt::mix(momentum, momentumTrack, param.lrate);

}

void erode_debris (
  silt::tensor_t<float> height,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentumTrack,
  silt::tensor_t<float> mass,
  silt::tensor_t<float> massTrack,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param
) {

  // velocity field?
  const float A = scale.x * scale.y;

  silt::set(massTrack, A);
  silt::set(momentumTrack, 0.0f);

  __erode_debris<<<block(rng.elem(), 512), 512>>>(
    height,
    mass,
    massTrack,
    momentum,
    momentumTrack,
    rng, height.shape(), scale, param);

  silt::mix(mass, massTrack, param.lrate);
  silt::mix(momentum, momentumTrack, param.lrate);

}

}


namespace soil {
using namespace silt;

/*
//
// Uplift Application
//  Note: Also applies erosion to map

__global__ void uplift(map_t map, const param_t param) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= map.elem)
    return;

  const vec3 scale = map.scale * 1E3f;    // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  const float transfer = map.transfer[n];
  const vec2 pos = __topos(map, n);
  __transfer(map, pos, transfer, Z);

  const float dt = param.timeStep;        //!< Geological Timestep [y] 
  const float uplift = param.uplift;      //!< Uplift Rate [m/y]
  const float mask = map.uplift[n];       //!< Uplift Mask

  map.height[n] += dt * mask * uplift / scale.z; //!< Total Height Delta

}

//
// Erosion Function
//

void erode(map_t& map, data_t& data, data_t& track, const param_t param, const size_t steps) {

  //
  // Initialize Rand-State Buffer (One Per Sample)
  //
  // note: the offset in the sequence should be number of times rand is sampled
  // that way the sampling procedure becomes deterministic

  const size_t n_samples = param.samples;
  if(map.rand.elem() != n_samples){
    map.rand = silt::tensor_t<curandState>(silt::shape(n_samples), silt::host_t::GPU);
    soil::seed(map.rand, 0, 4 * map.age);
  }

  if(map.transfer.elem() != map.elem){
    map.transfer = silt::tensor_t<float>(map.shape, silt::host_t::GPU);
  }

  const scale_t scale(map.scale);

  //
  // Execute Solution
  //

  for(size_t step = 0; step < steps; ++step){

    // Reset Estimates
    silt::set(track.discharge, 0.0f);
    silt::set(track.momentum, 0.0f);
    silt::set(track.mass, 0.0f);
    silt::set(track.debris, 0.0f);
    silt::set(track.debris_momentum, 0.0f);
    cudaDeviceSynchronize();

//    // Solve Estimates
    fluvial::solve<<<block(n_samples, 512), 512>>>(map, data, track, param, scale, n_samples);
//    debris::solve<<<block(n_samples, 512), 512>>>(map, data, track, n_samples, param);
    cudaDeviceSynchronize();

    // Filter Estimates
    silt::mix(data.momentum, track.momentum, param.lrate);
    silt::mix(data.discharge, track.discharge, param.lrate);
    silt::mix(data.mass, track.mass, param.lrate);
    silt::mix(data.debris, track.debris, param.lrate);
    silt::mix(data.debris_momentum, track.debris_momentum, param.lrate);
    cudaDeviceSynchronize();

    // Execute Height-Map Mass-Transfer
    silt::set(map.transfer, 0.0f);
    fluvial::mt<<<block(map.elem, 512), 512>>>(map, data, param, scale);
//    debris::mt<<<block(map.elem, 512), 512>>>(map, data, param);
    uplift<<<block(map.elem, 512), 512>>>(map, param);

    // Increment Model Age for Rand-State Initialization
    map.age++;

  }

}
*/

} // end of namespace soil



#endif