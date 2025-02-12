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

  size_t samples = 2048;
  size_t maxage = 1024;

  float settling = 1.0f;
  float maxdiff = 0.8f;
  float evapRate = 0.001f;
  float depositionRate = 0.05f;
  float entrainment = 0.25f;
  float gravity = 2.0f;
  float momentumTransfer = 1.0f;
  float minVol = 0.001f;
  float lrate = 0.01f;
  float exitSlope = 0.99f;
  float hscale = 0.1f;

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