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

__global__ void __transport_fluvial (
  silt::tensor_t<float> waterFlux,
  silt::tensor_t<float> massFlux,
  silt::view_t<silt::vec2> velocityFlux,
  silt::view_t<silt::vec3> albedoFlux,
  silt::tensor_t<silt::rng> rng,
  const silt::view_t<silt::vec2> layers,
  const silt::tensor_t<float> rainfall,
  const silt::tensor_t<float> discharge,
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
  const auto P = 1.0f / float(shape.elem);      //!< Sample Probability 
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
  const auto ks = param.suspensionRateFluvial;  //!< Fluvial Suspension Rate
  const auto kd = param.depositionRateFluvial;  //!< Fluvial Deposition Rate
  const auto fD = param.frictionFactor;         //!< Darcy-Weisbach Friction Factor []
  const auto alpha = param.fluvialExponent;     //!< Suspension Power               []
  const auto R = param.rainfall;                //!< Water Rainfall Rate            [m/y]

  // Trajectory Initialization
  const auto vel = velocity[ind];
  silt::vec2 grad = __grad(layers, shape, scale, pos, param.exitSlope);
  silt::vec2 speed = - (g * grad) + nu * vel;
  speed = speed / sqrtf(glm::length(L * speed));
  if(glm::length(speed) < eps)
    return;
    
  // Transport Source Terms
  const auto v = __length(vel);                             //!< [m/s]
  const auto shear = 0.125f * fD * rho_w * v * v;           //!< [kg/m/s^2]
  const auto power = __powf(shear * __length(grad), alpha);
  //  const auto power = __powf(discharge[ind], alpha) * __length(grad);

  const auto source_m = Q * ks * power;
  const auto source_w = Q * R * rainfall[ind];
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
    const auto accel = - (g * grad) + nu * velocity[ind];
    speed = (1.0f / (1.0f + dL * (tau + nu))) * speed + (dL / (1.0f + dL * (tau + nu))) * accel;

    // Transport Attenuation Update
    const auto decay_m = kd;// / (eps + R * rainfall[ind] + discharge[ind]);
    const auto decay_w = param.evapRate;
    const auto decay_v = 0.125f * fD / (eps + R * rainfall[ind] + discharge[ind]);

    att_m = att_m * __expf(-ds * decay_m);
    att_w = att_w * __expf(-ds * decay_w);
    att_v = att_v * __expf(-dL * decay_v);
    pos += v_step * v_unit;

  }

}

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
  const auto P = 1.0f / float(shape.elem);      //!< Sample Probability 
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
  silt::vec2 grad = __grad(layers, shape, scale, pos, param.exitSlope);
  silt::vec2 speed = - (g * grad);
  if(__length(speed) < eps)
    return;

  // Transport Source / Attenuation Terms
  const float excessSlope = (__length(grad) - theta);
  const float suspend = fmaxf(0.0f, kl * excessSlope - tau_y);

  const auto source_d = A * Q * suspend;
  const auto source_v = Q * (- g * grad + nu * velocity[ind]);
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
    const auto accel = - (g * grad) + nu * velocity[ind];
    speed = (1.0f / (1.0f + dL * (tau + nu))) * speed + (dL / (1.0f + dL * (tau + nu))) * accel;

    // Update Transport Attenuation
    const auto excessSlope = (__length(grad) - theta);                               //!< Local Excess Slope
    const auto excessStress = g * (excessSlope - tau_y / (att_d * source_d + eps));  //!< Shear Stress Balance
    const auto shearRate = (excessStress < 0.0f) ? kdd : kds;                        //!< Asymmetric Shear Rate
    const auto decay_d = - shearRate * excessStress;
    const auto decay_v = (nu + tau);

    att_d = att_d * __expf(-dL * decay_d);
    att_v = att_v * __expf(-dL * decay_v);
    pos += v_step * v_unit;

  }

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
  const silt::tensor_t<float> discharge,
  const silt::tensor_t<float> mass,
  const silt::const_view_t<silt::vec2> momentumFluvial,
  const silt::tensor_t<float> debris,
  const silt::const_view_t<silt::vec2> momentumDebris,
  silt::view_t<silt::vec3> albedo_bedrock,
  silt::view_t<silt::vec3> albedo_transport,
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
  const float kfs = param.suspensionRateFluvial;  // Fluvial Suspension Rate
  const float kfd = param.depositionRateFluvial;  // Fluvial Deposition Rate
  const float fD = param.frictionFactor;          // Darcy-Weisbach Friction Factor []
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
  const auto speed = momentumFluvial[n] - (g * grad);
  const auto v = __length(momentumFluvial[n]);                // Fluid Velocity           [m/s]
  const auto shear = 0.125f * fD * rho * v * v;               // Wall Shear Stress        [kg/m/s^2 = Pa]
  const auto power = __powf(shear * slope, alpha);            // Stream Power Function    [(kg/s^3)^a]
//  const auto power = __powf(discharge[n], alpha) * slope;
  const auto suspend = kfs * power;                           // Fluvial Suspension Rate  [m/y]

  const float deposit = kfd * mass[n];// / (eps + discharge[n]); // Fluvial Deposition Rate  [m/y]
  const float uplift = ku * upliftBase[n];                    // Terrain Uplift Rate      [m/y]

  // Debris Erosion Computation
  const float debrisHeight = debris[n];                                             // Debris Flow Height [m]
  const float excessSlope = (slope - param.critSlopeBedrock);                       // Excess Slope []
  const float shearLandslide = fmaxf(0.0f, kL * excessSlope - tau_y);               //
  const float shearYield = g * (debrisHeight * excessSlope - tau_y);                //
  const float suspendDebris = shearLandslide + kds * fmaxf(0.0f, shearYield);       // Debris Suspension Rate [m/y]
  const float depositDebris = fminf(debrisHeight, fmaxf(0.0f, -kdd * shearYield));  // Debris Deposition Rate [m/y]

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
  // Albedo Mixing Effect...
  //

  // Basically, if mass is larger than zero,
  //  than we want to add the mass to the
  const auto m = mass[n] + debris[n];

  if(layer.y == 0.0f) {

    albedo_surface[n] = albedo_bedrock[n];

  } else if(m > 0.0f && transfer > 0.0f) {

    const auto surface_color = albedo_surface[n];
    const auto transport_color = albedo_transport[n] / m;  //!< Surface Transport Color...

    // Mixing Rate: The mixture is weighted by the layer height and the amount of mass added,
    //  with the layer height limited by a maximum layer mixing depth.
    const auto w_surf = fminf(1.0f, layer.y * scale.z);
    const auto w_trsp = fmaxf(0.0f, transfer);
    const auto w = fmaxf(0.0f, w_trsp / (w_trsp + w_surf));
    const auto mix_color = w * transport_color + (1.0f - w) * surface_color;
    albedo_surface[n] = mix_color;

  }

}

//
// Kernel Launch Implementations
//

void soil::transport_fluvial (
  silt::tensor_t<float> layers,
  silt::tensor_t<float> rainfall,
  silt::tensor_t<float> discharge,
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

  const float A = scale.x * scale.y;
  const silt::shape shape = layers.shape();

  __transport_fluvial<<<block(rng.elem(), 512), 512>>> (
    waterFlux,
    massFlux,
    velocityFlux.view<silt::vec2>(),
    albedoFlux.view<silt::vec3>(),
    rng,
    layers.view<silt::vec2>(),
    rainfall,
    discharge,
    mass,
    momentum.view<silt::vec2>(),
    albedoSource.view<silt::vec3>(),
    shape, scale, param
  );

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

  const float A = scale.x * scale.y;
  const silt::shape shape = layers.shape();

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

}

void soil::mass_transfer (
  silt::tensor_t<float> delta,
  silt::tensor_t<float> layers,
  const silt::tensor_t<float> uplift,
  const silt::tensor_t<float> discharge,
  const silt::tensor_t<float> mass,
  const silt::tensor_t<float> momentumFluvial,
  const silt::tensor_t<float> debris,
  const silt::tensor_t<float> momentumDebris,
  silt::tensor_t<float> albedo_bedrock,
  silt::tensor_t<float> albedo_transport,
  silt::tensor_t<float> albedo_surface,
  const silt::vec3 scale,
  const soil::param_t param
) {

  const silt::shape shape = uplift.shape();
  
  __transfer<<<block(uplift.elem(), 512), 512>>> (
    delta.view<silt::vec2>(),
    layers.view<silt::vec2>(),
    uplift,
    discharge,
    mass,
    momentumFluvial.view<silt::vec2>(),
    debris,
    momentumDebris.view<silt::vec2>(),
    albedo_bedrock.view<silt::vec3>(),
    albedo_transport.view<silt::vec3>(),
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

__global__ void __agitation (
  silt::tensor_t<float> agitation,
  const silt::const_view_t<silt::vec2> delta,
  const silt::shape shape,
  const float decay,
  const float grow
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem)
    return;

  float ag = agitation[n];
  ag = (1.0f - decay) * ag;
  ag += grow * abs(delta[n].y);
  agitation[n] = ag;

}

void soil::agitation (
  silt::tensor_t<float> agitation,
  const silt::tensor_t<float> delta,
  const float decay,
  const float grow
) {

  const auto shape = agitation.shape();
  __agitation<<<block(shape.elem, 512), 512>>> (
    agitation,
    delta.view<silt::vec2>(),
    shape, decay, grow
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