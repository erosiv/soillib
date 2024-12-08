#ifndef SOILLIB_NODE_EROSION
#define SOILLIB_NODE_EROSION

#include <soillib/soillib.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

namespace soil {

// What do we have to do to get GPU Erosion Working?
// 1. Compute the Discharge Map from a Height-Map by Descending Particles based on Position
// 2. Introduce the Particle Volume, Make sure Discharge Works Based on Mass
// 3. Add Momentum to the Particles, Momentum Conservation
// 4. Introduce the Mass-Exchange Kernel
// 5. Introduce the Thermal Erosion Kernel

//
// Model Summarization
//

struct particle_t {

  particle_t(const size_t elem):elem(elem),
    pos(elem, soil::host_t::GPU),
    spd(elem, soil::host_t::GPU),
    vol(elem, soil::host_t::GPU),
    sed(elem, soil::host_t::GPU),
    slope(elem, soil::host_t::GPU){}

  const size_t elem;

  soil::buffer_t<vec2> pos;
  soil::buffer_t<vec2> spd;
  soil::buffer_t<float> vol;
  soil::buffer_t<float> sed;
  soil::buffer_t<float> slope;
};

struct model_t {

  model_t(soil::index index):
    index(index.as<soil::flat_t<2>>()),
    elem(index.elem()){}
  
  const size_t elem;
  const soil::flat_t<2> index;

  soil::buffer_t<float> height;
  soil::buffer_t<float> discharge;
  soil::buffer_t<vec2> momentum;
};

void gpu_erode(model_t& model, const size_t steps, const size_t maxage);

} // end of namespace soil

#endif