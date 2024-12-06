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

void gpu_erode(soil::buffer &buffer, soil::buffer& discharge, soil::buffer& momentum, const soil::index &index, const size_t steps, const size_t maxage);

} // end of namespace soil

#endif