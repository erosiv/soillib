#ifndef SOILLIB_NODE_EROSION
#define SOILLIB_NODE_EROSION

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <soillib/soillib.hpp>

#include <curand_kernel.h>

namespace soil {

//
// Model Summarization
//

struct param_t {

  size_t samples = 8192;
  size_t maxage = 128;
  float lrate = 0.2f;

  float timeStep = 10.0f; // [y]

  float rainfall = 1.5f;    // [m/y]
  float evapRate = 0.0001f; // [m^3/s]

  float gravity = 9.81f;   // [m/s]
  float viscosity = 0.05f; // [m^2/s]
  float bedShear = 0.025f; //

  float critSlope = 0.57f;
  float settleRate = 0.005f;
  float thermalRate = 0.005f;
  float debrisShear = 0.9f;

  float depositionRate = 0.01f;
  float suspensionRate = 0.0007f;
  float exitSlope = 0.0075f;
};

struct map_t {

  map_t(soil::index index, soil::vec3 scale):
    index(index.as<soil::flat_t<2>>()),
    scale(scale){}

  const soil::flat_t<2> index; // Buffer Indexing Structure
  const soil::vec3 scale;      // Value Scaling Factor (Real Coordinates)
  
  soil::buffer_t<float> height;
  soil::buffer_t<float> sediment;

};

//! data_t is a structure for storing the erosion model data
//! Effectively, this struct is a collection of buffers.
//! Note that this struct is agnostic to the map shape.
struct data_t {

  data_t(const size_t elem):
    elem{elem}{}

  const size_t elem;  //!< Total Buffer Elements
  int age = 0;        //!< Model Age

  soil::buffer_t<curandState> rand;

  soil::buffer_t<float> discharge;
  soil::buffer_t<float> discharge_track;

  soil::buffer_t<vec2> momentum;
  soil::buffer_t<vec2> momentum_track;

  soil::buffer_t<float> mass;
  soil::buffer_t<float> mass_track;

  soil::buffer_t<float> debris;
  soil::buffer_t<float> debris_track;

  soil::buffer_t<vec2> debris_momentum;
  soil::buffer_t<vec2> debris_momentum_track;

};

void erode(map_t& map, data_t &data, const param_t param, const size_t steps);

} // end of namespace soil

#endif