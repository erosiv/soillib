#ifndef SOILLIB_NODE_EROSION
#define SOILLIB_NODE_EROSION

#include <soillib/soillib.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <curand_kernel.h>

namespace soil {

//
// Model Summarization
//

struct param_t {

  size_t samples = 8192;
  size_t maxage = 128;
  float lrate = 0.2f;

  float timeStep = 10.0f;   // [y]

  float rainfall = 1.5f;    // [m/y]
  float evapRate = 0.0001f; // [m^3/s]

  float gravity = 9.81f;    // [m/s]
  float viscosity = 0.03f;  // [m^2/s]

  float critSlope = 0.57f;
  float settleRate = 0.005f;
  float thermalRate = 0.005f;
  
  float depositionRate = 0.01f;
  float suspensionRate = 0.0007f;
  float exitSlope = 0.01f;
  
};

struct model_t {

  model_t(soil::index index, soil::vec3 scale):
    index(index.as<soil::flat_t<2>>()),
    scale(scale),
    elem(index.elem()){}
  
  const size_t elem;            // Total Buffer Elements
  const soil::flat_t<2> index;  // Buffer Indexing Structure
  const soil::vec3 scale;       // Value Scaling Factor (Real Coordinates)

  int age = 0;

  soil::buffer_t<float> height;
  soil::buffer_t<float> sediment;

  soil::buffer_t<float> discharge;
  soil::buffer_t<float> discharge_track;

  soil::buffer_t<vec2> momentum;
  soil::buffer_t<vec2> momentum_track;

  soil::buffer_t<curandState> rand;

};

void erode(model_t& model, const param_t param, const size_t steps);

} // end of namespace soil

#endif