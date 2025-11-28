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

  size_t maxage = 512;
  float lrate = 0.1f;

  float timeStep = 10.0f; //!< [y]

  float gravity = 9.81f;    //!< [m/s^2]
  float uplift = 0.1f;      //!< [m/y]
  float rainfall = 1.0f;    //!< [m/y]
  float evapRate = 0.0001f; //!< [m^3/s]

  // Fluvial Erosion Parameters
  float viscosity = 0.000001f;      //!< [m^2/s]
  float bedShear = 12.5f;           //!< [Pa]
  float fluvialDensity = 1000.0f;   //!< [kg/m^3]
  float frictionFactor = 0.02f;     //!< []
  float fluvialExponent = 0.01f;
  float suspensionRate = 0.0000008f; //!<
  float depositionRate = 0.00001f;
  float exitSlope = 0.025f;

  // Thermal Erosion Parameters
  float critSlope = 0.57f;
  float debrisCreepRate = 0.0025f;
  float debrisSuspensionRate = 0.00025f;
  float debrisDepositionRate = 0.000025f;
  float debrisYieldStress = 2E6;  //!< Pa*s
  float debrisViscosity = 0.001f;
  float debrisBedShear = 0.0125f;
  float debrisDensity = 2.0f;  //!< [kg/m^3]
  
  // Arbitrary Body Force
  glm::vec2 force = glm::vec2(0.0f);

};

//
// Unified Erosion Kernels
//

void erode (
  silt::tensor_t<float> height,
  silt::tensor_t<float> velocity,
  silt::tensor_t<float> velocity_track,
  silt::tensor_t<float> discharge,
  silt::tensor_t<float> discharge_track,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param
);

void erode_debris (
  silt::tensor_t<float> height,
  silt::tensor_t<float> velocity,
  silt::tensor_t<float> velocity_track,
  silt::tensor_t<float> mass,
  silt::tensor_t<float> mass_track,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param
);

} // end of namespace soil

#endif