#ifndef SOILLIB_MODEL_EROSION
#define SOILLIB_MODEL_EROSION

#include <soillib/soillib.hpp>
#include <silt/core/tensor.hpp>
#include <curand_kernel.h>

// Soillib Erosion Model
//
// This file contains the high-level data types.
// ... description ...

namespace soil {

//! param_t contains the full physical
//! parameterization of the erosion model.
struct param_t {

  // Simulation Parameters
  size_t maxage = 512;
  float lrate = 0.1f;
  float timeStep = 10.0f; //!< [y]

  // Boundary Conditions  
  float exitSlope = 0.025f;
  float uplift = 0.1f;      //!< [m/y]
  float rainfall = 1.0f;    //!< [m/y]
  float gravity = 9.81f;    //!< [m/s^2]

  float evapRate = 0.0001f; //!< [m^3/s]

  float critSlope = 0.57f;  
  float frictionFactor = 0.02f;     //!< []
  float fluvialExponent = 0.01f;
  float suspensionRate = 0.0000008f; //!<
  float depositionRate = 0.00001f;

  // Thermal Erosion Parameters
  float debrisSuspensionRate = 0.00025f;
  float debrisDepositionRate = 0.000025f;
  float debrisViscousStress = 0.001f;
  float debrisYieldStress = 2E6;  //!< Pa*s
  
  // Arbitrary Body Force
  glm::vec2 force = glm::vec2(0.0f);

};

//! Parameters for Momentum Conservation
struct momentum_param_t {
  float density = 1000.0f;  //!< [g/m^3]
  float viscosity = 0.5f;   //!< []
  float bedShear = 0.1f;    //!< []
};

using layer_t = silt::vec2;

//
// Unified Erosion Kernels
//

void transport_fluvial (
  silt::tensor_t<float> layers,
  silt::tensor_t<float> discharge,
  silt::tensor_t<float> discharge_track,
  silt::tensor_t<float> mass,
  silt::tensor_t<float> mass_track,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentum_track,
  silt::tensor_t<float> albedo_bedrock,
  silt::tensor_t<float> albedo_transport,
  silt::tensor_t<float> albedo_surface,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
);

void transport_debris (
  silt::tensor_t<float> layers,
  silt::tensor_t<float> velocity,
  silt::tensor_t<float> velocity_track,
  silt::tensor_t<float> mass,
  silt::tensor_t<float> mass_track,
  silt::tensor_t<float> albedo_bedrock,
  silt::tensor_t<float> albedo_transport,
  silt::tensor_t<float> albedo_surface,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
);

void mass_transfer (
  silt::tensor_t<float> delta,
  silt::tensor_t<float> layers,
  const silt::tensor_t<float> uplift,
  const silt::tensor_t<float> discharge,
  const silt::tensor_t<float> mass,
  const silt::tensor_t<float> momentum,
  const silt::tensor_t<float> debris,
  const silt::tensor_t<float> momentumDebris,
  silt::tensor_t<float> albedo_bedrock,
  silt::tensor_t<float> albedo_transport,
  silt::tensor_t<float> albedo_surface,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
);

void mass_creep (
  silt::tensor_t<float> delta,
  const silt::tensor_t<float> layers,
  const silt::vec3 scale,
  const soil::param_t param
);

void layer_merge (
  silt::tensor_t<float> height,
  const silt::tensor_t<float> layers
);

void layer_albedo (
  silt::tensor_t<float> albedo,
  const silt::tensor_t<float> layers,
  const float ext_sediment,
  const silt::tensor_t<float> colorA,
  const silt::tensor_t<float> colorB,
  const silt::tensor_t<float> discharge,
  const float ext_discharge,
  const silt::vec3 colorW
);

} // end of namespace soil

#endif