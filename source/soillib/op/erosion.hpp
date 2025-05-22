#ifndef SOILLIB_NODE_EROSION
#define SOILLIB_NODE_EROSION

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <soillib/soillib.hpp>

#include <soillib/op/rbf.hpp>
#include <soillib/index/kdtree.hpp>

#include <curand_kernel.h>

namespace soil {

//
// Model Summarization
//

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
  float bedShear = 12.5;            //!< [Pa]
  float fluvialDensity = 1000.0f;   //!< [kg/m^3]
  float frictionFactor = 0.02f;       //!< 
  float fluvialExponent = 0.01f;
  float depositionRate = 0.1f;
  float suspensionRate = 0.05f;
  float exitSlope = 0.0075f;

  // Thermal Erosion Parameters
  float critSlope = 0.57f;
  float debrisCreepRate = 0.1f;
  float debrisSuspensionRate = 0.1f;
  float debrisDepositionRate = 0.0025f;
  float debrisShear = 0.01f;
  float debrisViscosity = 0.001f;
  float debrisBedShear = 0.9f;
  float debrisDensity = 2.0f;  //!< [kg/m^3]
  

};

struct map_grid {

  map_grid(soil::index index, soil::vec3 scale):
    index(index.as<soil::flat_t<2>>()),
    scale(scale),
    elem(index.elem()){}

  const size_t elem;            // Total Number of Elements
  const soil::flat_t<2> index;  // Buffer Indexing Structure
  const soil::vec3 scale;       // Value Scaling Factor (Real Coordinates)

  soil::buffer_t<float> height;
  soil::buffer_t<float> sediment;

  // User Control Fields
  soil::buffer_t<float> uplift;   //!< Uplift Control Map
  soil::buffer_t<float> rainfall; //!< Rainfall Control Map

  soil::buffer_t<float> transfer; //!< Transferred Material

  soil::buffer_t<curandState> rand;
  int age = 0;                  //!< Model Age

};

using map_t = map_grid;

//! data_t is a structure for storing the erosion model data
//! Effectively, this struct is a collection of buffers.
//! Note that this struct is agnostic to the map shape.
struct data_t {

  data_t():elem{0}{}
  data_t(const size_t elem):
    elem{elem}{
    this->mass            = soil::buffer_t<float>(this->elem, soil::host_t::GPU);
    this->discharge       = soil::buffer_t<float>(this->elem, soil::host_t::GPU);
    this->momentum        = soil::buffer_t<vec2>(this->elem, soil::host_t::GPU);
    this->debris          = soil::buffer_t<float>(this->elem, soil::host_t::GPU);
    this->debris_momentum = soil::buffer_t<vec2>(this->elem, soil::host_t::GPU);
  }

  const size_t elem;  //!< Total Buffer Elements

  soil::buffer_t<float> discharge;
  soil::buffer_t<vec2> momentum;
  soil::buffer_t<float> mass;
  soil::buffer_t<float> debris;
  soil::buffer_t<vec2> debris_momentum;

};

void erode(map_grid& map, data_t &data, data_t &track, const param_t param, const size_t steps);

} // end of namespace soil

#endif