#ifndef SOILLIB_MODEL_GRAPH
#define SOILLIB_MODEL_GRAPH

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! Compute the Flow-Graph of Steepest-Neighbors for a Height-Field
silt::tensor_t<int> steepest(const silt::tensor_t<float> height);

} // end of namespace soil

#endif