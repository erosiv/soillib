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
  size_t maxage = 512;      //!< Maximum Particle Age
  float lrate = 1.0f;       //!< Learning Rate []
  float timeStep = 250.0f;  //!< Geological Timestep [y]

  // Boundary / Environmental Conditions  
  float exitSlope = 0.02f;  //!< Boundary Slope [m/m]
  float uplift = 0.001f;    //!< Uplift Rate [m/y]
  float rainfall = 1.0f;    //!< Rainfall Rate [m/y]
  float gravity = 9.81f;    //!< Specific Gravity [m/s^2]
  float evapRate = 0.0002f; //!< Water Evaporation Rate [m^3/s]

  // Erosion Parameters
  float frictionFactor = 0.06f;           //!< Sediment Friction Factor []
  float fluvialExponent = 2.0f;           //!< Shear Stress Power Exponent

  float suspensionRateFluvial = 4.5E-8f;  //!< Suspension Rate Fluvial Erosion
  float depositionRateFluvial = 0.04f;    //!< Deposition Rate Fluvial Erosion

  float suspensionRateDebris = 0.001f;    //!< Suspension Rate Debris-Flow Erosion
  float depositionRateDebris = 0.01f;     //!< Deposition Rate Debris-Flow Erosion
  float landslideRateDebris = 0.003f;     //!< Suspension Rate Landslide Erosion

  // Material Properties
  float critSlopeBedrock = 0.57f;   //!< Critical Slope of Bedrock
  float critSlopeSediment = 0.3f;   //!< Critical Slope of Sediment
  float yieldStress = 0.001f;        //!< Bedrock Yield Stress [Pa*s]

  float viscosityWater = 1E-6f;     //!< [m^2/s]
  float bedShearWater = 0.0075f;    //!< [Pa*s]
  float densityWater = 1.0f;        //!< [kg/m^3]

  float viscosityDebris = 0.0f;     //!< [m^2/s]
  float bedShearDebris = 0.99f;     //!< [Pa*s]
  float densityDebris = 2.0f;       //!< [kg/m^3]

  // Arbitrary Body Force
  glm::vec2 force = glm::vec2(0.0f);

};

using layer_t = silt::vec2;

//
// Unified Erosion Kernels
//

// Transport Kernels:
//  Compute the transport quantity estimates.

void transport_fluvial (
  silt::tensor_t<float> layers,
  silt::tensor_t<float> rainfall,
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
  const soil::param_t param
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
  const soil::param_t param
);

// Mass-Transfer Kernels:
//  Compute the total effect on the terrain
//  by computing the delta tensor.

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
  const soil::param_t param
);

void mass_creep (
  silt::tensor_t<float> delta,
  const silt::tensor_t<float> layers,
  const silt::vec3 scale,
  const soil::param_t param
);

// Additional Utility Kernels

void layer_merge (
  silt::tensor_t<float> height,
  const silt::tensor_t<float> layers
);

void agitation (
  silt::tensor_t<float> agitation,
  const silt::tensor_t<float> delta,
  const float decay,
  const float grow
);

//
// Albedo Generating Functions
//

void albedo_layer (
  silt::tensor_t<float> albedo,
  const silt::tensor_t<float> albedoBedrock,
  const silt::tensor_t<float> albedoSediment,
  const silt::tensor_t<float> layers,
  const float scaleSediment,
  const silt::vec3 shiftSediment
);

void albedo_discharge (
  silt::tensor_t<float> albedo,
  const silt::tensor_t<float> discharge,
  const silt::vec3 colorDischarge,
  const float extinction,
  const float scale
);

// void layer_albedo (
//   // Tensor Inputs
// 
//   const silt::tensor_t<float> layers,
//   
//   const silt::tensor_t<float> agitation,
//   // Parameters
//   const float scaleSediment,
//   const float scaleDischarge,
//   const float scaleAgitation,
//   const silt::vec3 shiftSediment,
//   const silt::vec3 shiftDischarge,
//   const silt::vec3 shiftAgitation
// );

} // end of namespace soil

#endif