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

  float exitSlope = 0.025f;
  float critSlope = 0.57f;

  float uplift = 0.1f;      //!< [m/y]
  float rainfall = 1.0f;    //!< [m/y]
  float evapRate = 0.0001f; //!< [m^3/s]

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
  float gravity = 9.81f;    //!< [m/s^2]
  float density = 1000.0f;  //!< [g/m^3]
  float viscosity = 0.5f;   //!< []
  float bedShear = 0.1f;    //!< []
};

//
// Unified Erosion Kernels
//

void erode (
  silt::tensor_t<float> height,
  silt::tensor_t<float> discharge,
  silt::tensor_t<float> discharge_track,
  silt::tensor_t<float> mass,
  silt::tensor_t<float> mass_track,
  silt::tensor_t<float> momentum,
  silt::tensor_t<float> momentum_track,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
);

void erode_debris (
  silt::tensor_t<float> height,
  silt::tensor_t<float> velocity,
  silt::tensor_t<float> velocity_track,
  silt::tensor_t<float> mass,
  silt::tensor_t<float> mass_track,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
);

} // end of namespace soil

#endif