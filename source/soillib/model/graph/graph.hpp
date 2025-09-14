#ifndef SOILLIB_MODEL_GRAPH
#define SOILLIB_MODEL_GRAPH

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! Compute the Direction of the FLow
silt::tensor_t<int> direction(const silt::tensor_t<float> height);

//! Compute the Flow-Graph of Steepest-Neighbors for a Height-Field
silt::tensor_t<int> steepest(const silt::tensor_t<float> height);

//! Compute the Upstream Accumulation of a Field
silt::tensor_t<float> accumulate(const silt::tensor_t<int> graph, const silt::tensor_t<float> field, const size_t iter);

} // end of namespace soil

#endif