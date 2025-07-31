#ifndef SOILLIB_MODEL_EROSION
#define SOILLIB_MODEL_EROSION

#include <soillib/soillib.hpp>
#include <soillib/core/tensor.hpp>
#include <curand_kernel.h>

// Soillib Erosion Model
//
// This file contains the high-level data types.
// ... description ...

namespace soil {

//! param_t contains the full physical
//! parameterization of the erosion model.
struct param_t {

  size_t samples = 8192;
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

//! data_t stores the tensors of transported quantities.
struct data_t {

  data_t():elem{0}{}
  data_t(const soil::shape _shape):
    shape(_shape),
    elem{_shape.elem}{
//    this->mass            = soil::tensor_t<float>(this->shape, soil::host_t::GPU);
//    this->discharge       = soil::tensor_t<float>(this->shape, soil::host_t::GPU);
//    this->momentum        = soil::tensor_t<vec2>(this->shape, soil::host_t::GPU);
//    this->debris          = soil::tensor_t<float>(this->shape, soil::host_t::GPU);
//    this->debris_momentum = soil::tensor_t<vec2>(this->shape, soil::host_t::GPU);
  }

  const soil::shape shape;
  const int elem;  //!< Total Buffer Elements

  soil::tensor_t<float> discharge;
  soil::tensor_t<float> momentum;         //!< Note: Require different shape!
  soil::tensor_t<float> mass;
  soil::tensor_t<float> debris;
  soil::tensor_t<float> debris_momentum;

};

struct scale_t {
  scale_t(const soil::vec3 scale){
    this->x = scale.x * 1E3f; //!< Scale to m <- km
    this->y = scale.y * 1E3f; //!< Scale to m <- km
    this->z = scale.z * 1E3f; //!< Scale to m <- km
    this->Ac = x*y;
    this->Vc = x*y*z;
    this->cl = soil::vec2(x, y);
    this->len = glm::length(cl);
  };
  float x;
  float y;
  float z;
  float Ac;
  float Vc;
  soil::vec2 cl;
  float len;
};

struct map_t {

  map_t(soil::shape shape, soil::vec3 scale):
    shape(shape),
    scale(scale),
    elem(shape.elem){}

  const size_t elem;            // Total Number of Elements
  const soil::shape shape;      // Buffer Indexing Structure
  const soil::vec3 scale;       // Value Scaling Factor (Real Coordinates)

  soil::tensor_t<float> height;
  soil::tensor_t<float> sediment;

  // User Control Fields
  soil::tensor_t<float> uplift;   //!< Uplift Control Map
  soil::tensor_t<float> rainfall; //!< Rainfall Control Map

  soil::tensor_t<float> transfer; //!< Transferred Material

  soil::tensor_t<curandState> rand;
  int age = 0;                  //!< Model Age

};

void erode(map_t& map, data_t &data, data_t &track, const param_t param, const size_t steps);

} // end of namespace soil

#endif