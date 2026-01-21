#ifndef SOILLIB_MODEL_EROSION_CU
#define SOILLIB_MODEL_EROSION_CU
#define HAS_CUDA

#include <soillib/soillib.hpp>

#include <silt/core/error.hpp>
#include <silt/core/tensor.hpp>
#include <silt/op/gather.hpp>
#include <silt/op/common.hpp>

#include <soillib/model/path/erosion.hpp>
#include <soillib/model/path/erosion_map.cu>

#include <math_constants.h>

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

//
// Fluvial Erosion Transport and Normalization
//

__global__ void __transport_fluvial (
  silt::tensor_t<float> waterFlux,
  silt::tensor_t<float> massFlux,
  silt::view_t<silt::vec2> velocityFlux,
  silt::view_t<silt::vec3> albedoFlux,
  silt::tensor_t<silt::rng> rng,
  const silt::view_t<silt::vec2> layers,
  const silt::tensor_t<float> waterSource,
  const silt::tensor_t<float> waterHeight,
  const silt::tensor_t<float> mass,
  const silt::view_t<silt::vec2> velocity,
  const silt::view_t<silt::vec3> albedoSource,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= rng.elem()) return;

  // Scaled Sampling Procedure
  const auto A = scale.x * scale.y;             //!< Cell Area                      [m^2]
  const auto L = silt::vec2(scale.x, scale.y);  //!< Cell Width                     [m]
  const auto N = rng.elem();                    //!< Sample Count
  const auto P = 1.0f / float(A * shape.elem);  //!< Sample Probability 
  const auto Q = 1.0f / (P * N);                //!< Normalization Factor
  const auto eps = 1E-12f;                      //!< Numerical Threshold
  auto pos = silt::vec2 {                       //!< Sample Position
    0.5f + curand_uniform(&rng[n])*float(shape[0] - 1),
    0.5f + curand_uniform(&rng[n])*float(shape[1] - 1)
  };
  auto ind = __flatten(shape, pos);             //!< Sample Index

  // Physical Parameter Set
  const auto rho_w = param.densityWater;        //!< Density of Water               [kg/m^3]
  const auto rho_s = param.densityDebris;       //!< Density of Sediment            [kg/m^3]
  const auto tau = param.bedShearWater;         //!<
  const auto nu = param.viscosityWater;         //!< Kinematic Viscosity            [m^2/s]
  const auto g = param.gravity;                 //!< Gravitational Acceleration     [m/s^2]
  const auto ks = param.suspensionRateFluvial / 64.0f;  //!< Fluvial Suspension Rate
  const auto kd = param.depositionRateFluvial * 1.33f;  //!< Fluvial Deposition Rate
  const auto fD = param.frictionFactor / 8.0f;         //!< Darcy-Weisbach Friction Factor []
  const auto alpha = param.fluvialExponent;     //!< Suspension Power               []
  const auto R = param.rainfall;                //!< Water Rainfall Rate            [m/y]

  // Trajectory Initialization
  const auto vel = velocity[ind];
  silt::vec2 grad = __grad(layers, shape, scale, pos, param.exitSlope);
  silt::vec2 speed = - (g * grad) + nu * vel + param.force;
  speed = speed / sqrtf(glm::length(L * speed));
  if(glm::length(speed) < eps)
    return;
    
  // Transport Source Terms
  const auto v = __length(vel);                             //!< [m/s]
  const auto shear = 0.125f * fD * rho_w * v * v;           //!< [kg/m/s^2]
  const auto power = __powf(shear * __length(grad), alpha);
  //  const auto power = __powf(discharge[ind], alpha) * __length(grad);

  const auto source_m = Q * ks * power;
  const auto source_w = Q * R * waterSource[ind];
  const auto source_v = Q * (- (g * grad) + nu * vel);
  const auto source_a = source_m * albedoSource[ind];

  // Attenuation Terms
  float att_w = 1.0f;
  float att_m = 1.0f;
  float att_v = 1.0f;

  // Integrate Trajectory
  int iter = 0;
  while(!__oob(shape, pos) && ++iter < param.maxage) {

    // Tracking Step
    const int nind = __flatten(shape, pos);
    if(nind != ind) {
      ind = nind;
      atomicAdd(&waterFlux[ind],      att_w * source_w);
      atomicAdd(&massFlux[ind],       att_m * source_m);
      atomicAdd(&velocityFlux[ind].x, att_v * source_v.x);
      atomicAdd(&velocityFlux[ind].y, att_v * source_v.y);
      atomicAdd(&albedoFlux[ind].x,   att_m * source_a.x);
      atomicAdd(&albedoFlux[ind].y,   att_m * source_a.y);
      atomicAdd(&albedoFlux[ind].z,   att_m * source_a.z);
    }

    // Dynamic Timestep
    const auto v_norm = __length(speed);
    const auto v_unit = speed / v_norm;
    const auto v_step = __stepsize(pos, v_unit);
    const auto dL = v_step * __length(L);
    const auto ds = dL / v_norm;
    if(v_norm < eps)
      break;

    // Velocity Update (Implicit Euler)
    grad = __grad(layers, shape, scale, pos, param.exitSlope);
    const auto accel = - (g * grad) + nu * velocity[ind] + param.force;
    speed = (1.0f / (1.0f + dL * (tau + nu))) * speed + (dL / (1.0f + dL * (tau + nu))) * accel;

    // Transport Attenuation Update
    const auto decay_m = kd;// / (eps + waterHeight);
    const auto decay_w = param.evapRate;
    const auto decay_v = 0.125f * fD / (eps + waterHeight[ind]);

    att_m = att_m * __expf(-ds * decay_m);
    att_w = att_w * __expf(-ds * decay_w);
    att_v = att_v * __expf(-dL * decay_v);
    pos += v_step * v_unit;

  }

}

__global__ void __normalize_fluvial (
  const silt::tensor_t<float> waterFlux,
  const silt::tensor_t<float> massFlux,
  const silt::view_t<silt::vec2> velocityFlux,
  silt::view_t<silt::vec3> albedoFlux,
  const silt::tensor_t<silt::rng> rng,
  const silt::view_t<silt::vec2> layers,
  const silt::tensor_t<float> waterSource,
  silt::tensor_t<float> waterHeight,
  silt::tensor_t<float> mass,
  silt::view_t<silt::vec2> velocity,
  const silt::view_t<silt::vec3> albedoSource,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem) return;

  const auto A = (scale.x * scale.y);                 //!< Cell Area [m^2]
  const auto L = silt::vec2(scale.x, scale.y);        //!< Cell Width [m]
  const auto v = silt::vec2(1.0, 0.0);                //!< Velocity [m/s] (FIX)
  const auto norm = abs(v.x * L.y) + abs(v.y * L.x);  //!< [m^2/s]
  const auto pos = shape.unflatten(n);
  const auto grad = __grad(layers, shape, scale, pos, param.exitSlope); // []

  const auto m = massFlux[n];
  const auto a = albedoFlux[n];

  const auto source_w = param.rainfall * waterSource[n];
  const auto source_v = - param.gravity * grad + param.force;
  const auto source_m = 0.0f;

  waterHeight[n]  = (A * source_w + waterFlux[n]) / norm;
  mass[n]         = (A * source_m + m) / norm;
  velocity[n]     = (A * source_v + velocityFlux[n]) / norm;

  if(m > 0.0f && __length(a) > 0.0f) {
    albedoFlux[n] = a / m;
  } else {
    albedoFlux[n] = albedoSource[n];
  }

}

void soil::transport_fluvial (
  silt::tensor_t<float> layers,
  silt::tensor_t<float> rainfall,
  silt::tensor_t<float> waterHeight,
  silt::tensor_t<float> waterFlux,
  silt::tensor_t<float> mass,
  silt::tensor_t<float> massFlux,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> velocityFlux,
  silt::tensor_t<float> albedo_bedrock,
  silt::tensor_t<float> albedoFlux,
  silt::tensor_t<float> albedoSource,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const auto shapeIn = layers.shape();
  const auto shape = silt::shape(shapeIn[0], shapeIn[1]);

  __transport_fluvial<<<block(rng.elem(), 512), 512>>> (
    waterFlux,
    massFlux,
    velocityFlux.view<silt::vec2>(),
    albedoFlux.view<silt::vec3>(),
    rng,
    layers.view<silt::vec2>(),
    rainfall,
    waterHeight,
    mass,
    momentum.view<silt::vec2>(),
    albedoSource.view<silt::vec3>(),
    shape, scale, param
  );

  __normalize_fluvial<<<block(shape.elem, 512), 512>>> (
    waterFlux,
    massFlux,
    velocityFlux.view<silt::vec2>(),
    albedoFlux.view<silt::vec3>(),
    rng,
    layers.view<silt::vec2>(),
    rainfall,
    waterHeight,
    mass,
    momentum.view<silt::vec2>(),
    albedoSource.view<silt::vec3>(),
    shape, scale, param
  );

}

//
// Debris Flow Erosion Transport and Normalization
//

__global__ void __transport_debris (
  silt::tensor_t<float> massFlux,
  silt::view_t<silt::vec2> velocityFlux,
  silt::view_t<silt::vec3> albedoFlux,
  silt::tensor_t<silt::rng> rng,
  const silt::view_t<silt::vec2> layers,
  const silt::tensor_t<float> mass,
  const silt::view_t<silt::vec2> velocity,
  const silt::view_t<silt::vec3> albedoSource,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= rng.elem()) return;

  // Scaled Sampling Procedure
  const auto A = scale.x * scale.y;             //!< Cell Area                      [m^2]
  const auto L = silt::vec2(scale.x, scale.y);  //!< Cell Width                     [m]
  const auto N = rng.elem();                    //!< Sample Count
  const auto P = 1.0f / float(A * shape.elem);  //!< Sample Probability 
  const auto Q = 1.0f / (P * N);                //!< Normalization Factor
  const auto eps = 1E-12f;                      //!< Numerical Threshold
  auto pos = silt::vec2 {                       //!< Sample Position
    0.5f + curand_uniform(&rng[n])*float(shape[0] - 1),
    0.5f + curand_uniform(&rng[n])*float(shape[1] - 1)
  };
  auto ind = __flatten(shape, pos);             //!< Sample Index

  // Physical Parameter Set
  const auto theta = param.critSlopeBedrock;    //!< Material Critical Slope  [m/m]
  const auto nu = param.viscosityDebris;
  const auto tau = param.bedShearDebris;
  const auto g = param.gravity;
  const auto kl = param.landslideRateDebris;
  const auto kdd = param.depositionRateDebris;
  const auto kds = param.suspensionRateDebris;
  const auto tau_y = param.yieldStress;

  // Trajectory Initialization
  const auto vel = velocity[ind];
  silt::vec2 grad = __grad(layers, shape, scale, pos, param.exitSlope);
  silt::vec2 speed = - (g * grad) + nu * vel;
  speed = speed / sqrtf(__length(L * speed));
  if(__length(speed) < eps)
    return;

  // Transport Source / Attenuation Terms
  const float excessSlope = (__length(grad) - theta);
  const float suspend = fmaxf(0.0f, kl * excessSlope);

  const auto source_d = Q * suspend;
  const auto source_v = Q * (- g * grad + nu * vel);  
  const auto source_a = source_d * albedoSource[ind];

  float att_d = 1.0f;
  float att_v = 1.0f;

  // Iterate over Number of Steps
  int iter = 0;
  while(!__oob(shape, pos) && ++iter < param.maxage) {

    // Tracking Step
    const int nind = shape.flatten(pos);
    if(nind != ind) {
      ind = nind;
      atomicAdd(&massFlux[ind],       att_d * source_d);
      atomicAdd(&velocityFlux[ind].x, att_v * source_v.x);
      atomicAdd(&velocityFlux[ind].y, att_v * source_v.y);
      atomicAdd(&albedoFlux[ind].x,   att_d * source_a.x);
      atomicAdd(&albedoFlux[ind].y,   att_d * source_a.y);
      atomicAdd(&albedoFlux[ind].z,   att_d * source_a.z);
    }

    // Dynamic Timestep
    const auto v_norm = __length(speed);
    const auto v_unit = speed / v_norm;
    const auto v_step = __stepsize(pos, v_unit);
    const auto dL = v_step * __length(L);
    const auto ds = dL / v_norm;
    if(v_norm < eps)
      break;

    // Velocity Update (Implicit Euler)
    grad = __grad(layers, shape, scale, pos, param.exitSlope);
    const auto debrisHeight = eps + att_d * source_d;
    const auto accel = - (g * grad) + nu * velocity[ind];
    const auto decay = nu + tau / debrisHeight;
    const auto w = 1.0f / (1.0f + dL * decay);
    speed = w * speed + w * dL * accel;

    // Update Transport Attenuation
//    const float viscousStress = nu * v_len / debrisHeight / debrisHeight;
    const auto excessSlope = (__length(grad) - theta);                  //!< Local Excess Slope
    const auto excessStress = g * (excessSlope - tau_y / debrisHeight); //!< Shear Stress Balance
    const auto shearRate = (excessStress < 0.0f) ? kdd : kds;           //!< Asymmetric Shear Rate    
    const auto decay_d = ds * shearRate * excessStress / v_norm;
    const auto decay_v = (nu + tau / debrisHeight);

    att_d = att_d * __expf(decay_d);        //!< Attenuation Update Debris
    att_v = att_v * __expf(-dL * decay_v);  //!< Attenuation Update Velocity
    pos += v_step * v_unit;                 //!< Grid Position Update

  }

}

__global__ void __normalize_debris (
  const silt::tensor_t<float> massFlux,
  const silt::view_t<silt::vec2> velocityFlux,
  silt::view_t<silt::vec3> albedoFlux,
  const silt::tensor_t<silt::rng> rng,
  const silt::view_t<silt::vec2> layers,
  silt::tensor_t<float> mass,
  silt::view_t<silt::vec2> velocity,
  const silt::view_t<silt::vec3> albedoSource,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem) return;

  const auto A = (scale.x * scale.y);                 //!< Cell Area [m^2]
  const auto L = silt::vec2(scale.x, scale.y);        //!< Cell Width [m]
  const auto v = silt::vec2(1.0, 0.0);                //!< Velocity [m/s] (FIX)
  const auto norm = abs(v.x * L.y) + abs(v.y * L.x);  //!< [m^2/s]
  const auto pos = shape.unflatten(n);
  const auto grad = __grad(layers, shape, scale, pos, param.exitSlope); // []

  const auto m = massFlux[n];
  const auto a = albedoFlux[n];

  const auto source_v = - param.gravity * grad;
  const auto source_d = 0.0f;
  const auto source_a = 0.0f;

  mass[n]       = (A * source_d + m) / norm;
  velocity[n]   = (A * source_v + velocityFlux[n]) / norm;

  if(m > 0.0f && __length(a) > 0.0f) {
    albedoFlux[n] = a / m;
  } else {
    albedoFlux[n] = albedoSource[n];
  }

}

void soil::transport_debris (
  silt::tensor_t<float> layers,
  silt::tensor_t<float> velocity,
  silt::tensor_t<float> velocityFlux,
  silt::tensor_t<float> mass,
  silt::tensor_t<float> massFlux,
  silt::tensor_t<float> albedo_bedrock,
  silt::tensor_t<float> albedoFlux,
  silt::tensor_t<float> albedoSource,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const auto shapeIn = layers.shape();
  const auto shape = silt::shape(shapeIn[0], shapeIn[1]);

  __transport_debris<<<block(rng.elem(), 512), 512>>> (
    massFlux,
    velocityFlux.view<silt::vec2>(),
    albedoFlux.view<silt::vec3>(),
    rng,
    layers.view<silt::vec2>(),
    mass,
    velocity.view<silt::vec2>(),
    albedoSource.view<silt::vec3>(),
    shape, scale, param
  );

  __normalize_debris<<<block(shape.elem, 512), 512>>> (
    massFlux,
    velocityFlux.view<silt::vec2>(),
    albedoFlux.view<silt::vec3>(),
    rng,
    layers.view<silt::vec2>(),
    mass,
    velocity.view<silt::vec2>(),
    albedoSource.view<silt::vec3>(),
    shape, scale, param
  );

}

//! Mass-Transfer Execution Kernel
//! This kernel edits the height-field and material distributions.
//!
//! Dimensionalization Notes:
//!   - The height-field is stored as a dimensionless quantity, and
//!     can be converted to meters using the scale.z parameter.
//!   - The transported fields have to add the source terms explicitly,
//!     because not every position generates a sample. This is accounted
//!     for in the transport kernels.
//!
//! Basically, there will be some limiting velocity based on the parameters,
//!   which will depend on the slope. This in turn will lead to some kind of
//!   steady-state slope. We could model this directly. Some dimensional
//!   analysis could help in general with making the model more intuitive.
//!
__global__ void __transfer (
  silt::view_t<silt::vec2> deltas,
  const silt::view_t<silt::vec2> layers,
  const silt::tensor_t<float> upliftBase,
  const silt::tensor_t<float> waterHeight,
  const silt::tensor_t<float> mass,
  const silt::const_view_t<silt::vec2> velocityFluvial,
  const silt::tensor_t<float> debris,
  const silt::const_view_t<silt::vec2> momentumDebris,
  silt::view_t<silt::vec3> albedo_bedrock,
  silt::view_t<silt::vec3> albedoFluxFluvial,
  silt::view_t<silt::vec3> albedoFluxDebris,
  silt::view_t<silt::vec3> albedo_surface,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  // General Dimensionalized Parameters
  const float dt = param.timeStep;                // Simulation Timestep            [y]
  const float ku = param.uplift;                  // Terrain Uplift Rate            [m/y]
  const float kfs = param.suspensionRateFluvial / 64.0f;  // Fluvial Suspension Rate
  const float kfd = param.depositionRateFluvial * 1.33f;  // Fluvial Deposition Rate
  const float fD = param.frictionFactor / 8.0f;          // Darcy-Weisbach Friction Factor []
  const float alpha = param.fluvialExponent;      // Power Law Exponent             []
  const float rho = param.densityWater;           // Fluid Density                  [kg/m^3]
  const float g = param.gravity;                  // Gravitational Acceleration     [m/s^2]
  const float tau_y = param.yieldStress;          // Normalized Yield Stress
  const float kds = param.suspensionRateDebris;   // Debris Suspension Rate
  const float kdd = param.depositionRateDebris;   // Debris Deposition Rate
  const float kL = param.landslideRateDebris;     // Landslide Erosion Rate
  const float eps = 1E-12f;

  // Compute Local Properties
  const silt::vec2 pos = shape.unflatten(n);
  const silt::vec2 grad = __grad(layers, shape, scale, pos, param.exitSlope); // []
  const float L = glm::length(glm::vec2(scale.x, scale.y));                   // [m]
  const float slope = glm::length(grad);                                      // []
  
  // Fluvial Erosion Computation
  const auto speed = velocityFluvial[n];
  const auto v = __length(speed);                             // Fluid Velocity           [m/s]
  const auto shear = 0.125f * fD * rho * v * v;               // Wall Shear Stress        [kg/m/s^2 = Pa]
  const auto power = __powf(shear * slope, alpha);            // Stream Power Function    [(kg/s^3)^a]
//  const auto power = __powf(discharge[n], alpha) * slope;
  const auto suspend = kfs * power;                           // Fluvial Suspension Rate  [m/y]

  const auto massHeight = mass[n];
  const auto deposit = kfd * massHeight;// / (eps + waterHeight[n]); // Fluvial Deposition Rate  [m/y]
  const auto uplift = ku * upliftBase[n];                    // Terrain Uplift Rate      [m/y]

  // Debris Erosion Computation
  const auto debrisHeight = debris[n];                                             // Debris Flow Height [m]
  const auto excessSlope = (slope - param.critSlopeBedrock);                       // Excess Slope []
  const auto shearLandslide = fmaxf(0.0f, kL * excessSlope);                       //
  const auto shearYield = g * (debrisHeight * excessSlope - tau_y);                //
  const auto suspendDebris = shearLandslide + kds * fmaxf(0.0f, shearYield);       // Debris Suspension Rate [m/y]
  const auto depositDebris = fminf(debrisHeight, fmaxf(0.0f, -kdd * shearYield));  // Debris Deposition Rate [m/y]

  // Height-Field Update (Stabilized):
  //  The erosion system is not permitted to generate a pit, because pits become self-reinforcing and numerically
  //  unstable by this model. The model assumes no pits. Therefore, we can use the local slope to limit the erosion
  //  rate. Physically, this can be interpreted as an exponential approach towards the steady-state.

  // Transfer Procedure (Read, Modify, Write):
  //  Sediment is always added to the top layer,
  //  while remove procedes from top to bottom.
  //  Uplift only affects the bedrock layer.

  float transfer = dt * (deposit - suspend + depositDebris - suspendDebris);
  transfer = fmaxf(transfer, -0.25f * L * slope); // Limit Suspension
  transfer = fminf(transfer,  0.25f * L * 0.3f);  // Limit Deposition

  auto layer = layers[n];
  auto delta = deltas[n];
  delta.x += dt * uplift / scale.z;
  delta.y += fmaxf(0.0f, transfer / scale.z);

  if(transfer < 0.0f) {

    // Limited Transfer
    auto limited = fmaxf(-layer.y * scale.z, transfer);
    delta.y += limited / scale.z;
    transfer -= limited;

    // Bedrock Transfer
    delta.x += transfer / scale.z;
  
  }
  
  deltas[n] = delta;

  //
  // Surface / Transport Albedo Mixing
  //

  const auto albedoFluvial = albedoFluxFluvial[n];
  const auto albedoDebris = albedoFluxDebris[n];
  const auto totalHeight = massHeight + debrisHeight;
  const auto mixDepth = 1.0f;

  if(layer.y == 0.0f) {
    albedo_surface[n] = albedo_bedrock[n];
  } else if (totalHeight > 0.0f && transfer > eps) {

    const auto wMass = fminf(massHeight / totalHeight, 1.0f);
    const auto colorTransport = __mmin(1.0f, wMass * albedoFluvial + (1.0f - wMass) * albedoDebris);
    const auto colorSurface = __mmin(1.0f, albedo_surface[n]);

    const auto wSurf = fminf(mixDepth, layer.y * scale.z);
    const auto wTrsp = fmaxf(eps, transfer);
    const auto w = fminf(wTrsp / (wTrsp + wSurf), 1.0f);
    const auto colorMix = w * colorTransport + (1.0f - w) * colorSurface;
    albedo_surface[n] = colorMix;

  }

}

void soil::mass_transfer (
  silt::tensor_t<float> delta,
  silt::tensor_t<float> layers,
  const silt::tensor_t<float> uplift,
  const silt::tensor_t<float> waterHeight,
  const silt::tensor_t<float> mass,
  const silt::tensor_t<float> velocityFluvial,
  const silt::tensor_t<float> debris,
  const silt::tensor_t<float> momentumDebris,
  silt::tensor_t<float> albedo_bedrock,
  silt::tensor_t<float> albedoFluxFluvial,
  silt::tensor_t<float> albedoFluxDebris,
  silt::tensor_t<float> albedo_surface,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const silt::shape shape = uplift.shape();
  
  __transfer<<<block(uplift.elem(), 512), 512>>> (
    delta.view<silt::vec2>(),
    layers.view<silt::vec2>(),
    uplift,
    waterHeight,
    mass,
    velocityFluvial.view<silt::vec2>(),
    debris,
    momentumDebris.view<silt::vec2>(),
    albedo_bedrock.view<silt::vec3>(),
    albedoFluxFluvial.view<silt::vec3>(),
    albedoFluxDebris.view<silt::vec3>(),
    albedo_surface.view<silt::vec3>(),
    shape, scale, param
  );

}

//
// Mass Creeping Implementation
//  We note that the laplacian really corresponds to a gradient of gradients...
//  So if we take each of these gradients individually and just cap them at their
//  natural limit, we should get a limited transfer laplacian.
//  So we compute the total amount transferred based in the individually limited
//  amount, and then apply that exponentially which should give us a stable transfer
//  function. The function has to be fully symmetric in order for the transport to
//  be mass conservative.

// Note:
//  Make sure to take into account the timestep and the critical slope...
//  We will implement this and then simply see what it do ...
//  

// Creep Transport Implementation:
//  We use a rate-limited laplacian, by applying a divergence of gradients.
//  In other words, we take the difference of gradients between the cell and
//  its neighbors and apply that, limiting by the amount of available mass.
//  The question is, if we cap it using the critical slope? Probably not.
__global__ void __mass_creep (
  silt::view_t<silt::vec2> delta,
  const silt::const_view_t<silt::vec2> layers,
  const silt::shape shape,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  // Compute Neighboring Index Set
  const silt::ivec2 ipos = shape.unflatten(n);
  const int i00 = n;
  const int in0 = shape.flatten(ipos + silt::ivec2(-1, 0));
  const int ip0 = shape.flatten(ipos + silt::ivec2( 1, 0));
  const int i0n = shape.flatten(ipos + silt::ivec2( 0,-1));
  const int i0p = shape.flatten(ipos + silt::ivec2( 0, 1));

  // Sample Neighboring Layer Configuration
  const auto l00 = layers[i00];
  const auto ln0 = shape.oob(ipos + silt::ivec2(-1, 0)) ? l00 : layers[in0];
  const auto lp0 = shape.oob(ipos + silt::ivec2( 1, 0)) ? l00 : layers[ip0];
  const auto l0n = shape.oob(ipos + silt::ivec2( 0,-1)) ? l00 : layers[i0n];
  const auto l0p = shape.oob(ipos + silt::ivec2( 0, 1)) ? l00 : layers[i0p];

  const auto h00 = (l00.x + l00.y) * scale.z;
  const auto hn0 = (ln0.x + ln0.y) * scale.z;
  const auto hp0 = (lp0.x + lp0.y) * scale.z;
  const auto h0n = (l0n.x + l0n.y) * scale.z;
  const auto h0p = (l0p.x + l0p.y) * scale.z;

  // Compute the Total Transfer Amount:
  //  We take the height-difference minus the critical slope,
  //  which would give us the excess until equilibriation.
  //  Then, we cut that in half due to the symmetry of what is
  //  transferred over. Finally, we divide the whole thing by
  //  four due to the four elements contributing at once, so
  //  that the entire thing is unconditionally stable.

  const float critSlope = param.critSlopeSediment;
  const auto __transfer = [critSlope, scale](const silt::vec2& lb, const silt::vec2& lt, const float dx) {
    const float hb = (lb.x + lb.y) * scale.z; // Height Bottom
    const float ht = (lt.x + lt.y) * scale.z; // Height Top
    const float tmax = 0.5f * ((ht - hb) - critSlope * dx);
    return fmaxf(0.0f, fminf(lt.y * scale.z, tmax));
  };

  float t = 0.0f;

  if(hp0 > h00) {
    t += __transfer(l00, lp0, scale.x);
  } else {
    t -= __transfer(lp0, l00, scale.x);
  }

  if(hn0 > h00) {
    t += __transfer(l00, ln0, scale.x);
  } else {
    t -= __transfer(ln0, l00, scale.x);
  }

  if(h0p > h00) {
    t += __transfer(l00, l0p, scale.y);
  } else {
    t -= __transfer(l0p, l00, scale.y);
  }

  if(h0n > h00) {
    t += __transfer(l00, l0n, scale.y);
  } else {
    t -= __transfer(l0n, l00, scale.y);
  }

  delta[n].y += 0.25f * t  / scale.z;

}

void soil::mass_creep (
  silt::tensor_t<float> delta,
  const silt::tensor_t<float> layers,
  const silt::vec3 scale,
  const soil::param_t param
) {
 
  const auto shapeIn = layers.shape();
  const auto shape = silt::shape(shapeIn[0], shapeIn[1]);
  __mass_creep<<<block(shape.elem, 512), 512>>> (
    delta.view<silt::vec2>(),
    layers.view<silt::vec2>(),
    shape, scale, param
  );

}

//
// Layer and Albedo Management
//

__global__ void __layer_merge (
  silt::tensor_t<float> height,
  const silt::const_view_t<silt::vec2> layers
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= height.elem())
    return;

  const auto layer = layers[n];
  height[n] = layer.x + layer.y;

}

void soil::layer_merge (
  silt::tensor_t<float> height,
  const silt::tensor_t<float> layers
) {

  __layer_merge<<<block(height.elem(), 512), 512>>> (
    height,
    layers.view<silt::vec2>()
  );

}

__global__ void __albedo_layer (
  silt::view_t<silt::vec3> albedo,
  const silt::const_view_t<silt::vec3> albedoBedrock,
  const silt::const_view_t<silt::vec3> albedoSediment,
  const silt::const_view_t<silt::vec2> layers,
  const float scaleSediment,
  const silt::vec3 shiftSediment,
  const silt::shape shape
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  const auto layer = layers[n];
  const auto colorBedrock = albedoBedrock[n];
  const auto colorSediment =  __mmin(1.0f, albedoSediment[n] + shiftSediment);

  const auto blend = 1.0f / (1.0f + scaleSediment * layer.y);
  albedo[n] = blend * colorBedrock + (1.0f - blend) * colorSediment;

//  // Discharge Color
//  //  const auto blendD = fmaxf(0.5f, 1.0f / (1.0f + extD * discharge[n]));  //!< Approaches Zero for High Discharge
//  //  const auto colorWater = colorW;
//  //  color = blendD * color + (1.0f - blendD) * colorWater;
//
//  // Shift the color depending on agitation
//  const float ag = extD * agitation[n];
//  const float shift = 0.2f * ag / sqrt(1 + ag * ag);
//  color = __mmin(1.0f, color * (1.0f + shift));
//  albedo[n] = color;

}

// Extinction Based Discharge Blending
__global__ void __albedo_stratum (
  silt::view_t<silt::vec3> albedoBedrock,
  const silt::tensor_t<float> uplift,
  const silt::const_view_t<silt::vec2> layers,
  const silt::vec3 scale,
  const soil::param_t param,
  const silt::vec3 colorA,
  const silt::vec3 colorB,
  const float age,
  const float freq,
  const silt::shape shape
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  // Total Uplift Displacement
  const auto shift = age * param.uplift * uplift[n];
  const auto layer = layers[n];
  const auto depth = fmaxf(shift - layer.x * scale.z, 0.0f);
  
  // The surface bedrock color is the total displacement,
  //  minus the current bedrock height.

  const int index = __floorf(depth / freq);
  if(index % 2 == 0){
    albedoBedrock[n] = colorA;// ... sample the color ...
  } else {
    albedoBedrock[n] = colorB;// ... sample the color ...
  }

}

void soil::albedo_stratum (
  silt::tensor_t<float> albedoBedrock,
  const silt::tensor_t<float> uplift,
  const silt::tensor_t<float> layers,
  const silt::vec3 scale,
  const soil::param_t param,
  const silt::vec3 colorA,
  const silt::vec3 colorB,
  const float age,
  const float freq
) {

  const auto shape = uplift.shape();
  __albedo_stratum<<<block(shape.elem, 512), 512>>> (
    albedoBedrock.view<silt::vec3>(),
    uplift,
    layers.view<silt::vec2>(),
    scale,
    param,
    colorA,
    colorB,
    age,
    freq,
    shape
  );

}

// Extinction Based Discharge Blending
__global__ void __albedo_discharge (
  silt::view_t<silt::vec3> albedo,
  const silt::tensor_t<float> discharge,
  const silt::vec3 colorDischarge,
  const float extinction,
  const float scale,
  const silt::shape shape
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  const auto color = albedo[n];
  const auto value = fmaxf(0.0f, discharge[n]);
  const auto blend = scale * (1.0f - __expf(-extinction * value));
  albedo[n] = blend * colorDischarge + (1.0f - blend) * color;

}

void soil::albedo_layer (
  silt::tensor_t<float> albedo,
  const silt::tensor_t<float> albedoBedrock,
  const silt::tensor_t<float> albedoSediment,
  const silt::tensor_t<float> layers,
  const float scaleSediment,
  const silt::vec3 shiftSediment
) {
  
  const auto shapeIn = albedo.shape();
  const auto shape = silt::shape(shapeIn[0], shapeIn[1]);
  __albedo_layer<<<block(shape.elem, 512), 512>>> (
    albedo.view<silt::vec3>(),
    albedoBedrock.view<silt::vec3>(),
    albedoSediment.view<silt::vec3>(),
    layers.view<silt::vec2>(),
    scaleSediment,
    shiftSediment,
    shape
  );

}

void soil::albedo_discharge (
  silt::tensor_t<float> albedo,
  const silt::tensor_t<float> discharge,
  const silt::vec3 colorDischarge,
  const float extinction,
  const float scale
) {
 
  const auto shapeIn = albedo.shape();
  const auto shape = silt::shape(shapeIn[0], shapeIn[1]);

  __albedo_discharge<<<block(shape.elem, 512), 512>>> (
    albedo.view<silt::vec3>(),
    discharge,
    colorDischarge,
    extinction, scale,
    shape
  );

}


#endif