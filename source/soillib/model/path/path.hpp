#ifndef SOILLIB_MODEL_PATH
#define SOILLIB_MODEL_PATH

#include <soillib/soillib.hpp>
#include <silt/core/tensor.hpp>
#include <soillib/model/path/erosion.hpp>

namespace soil {

// General Solver Procedure...
//  Basically, we should basically make sure that
//  the algorithm in its absolute core form works.
//  This means that the rng buffer is passed in here...
//  Additionally, all other fields are assumed provided.
//  Then we can run stability analysis on this.
//  We could make everything else more stable of course
//  if we don't integrate the momentum equation in the other
//  particles as well and instead assume a static velocity field.
//  We will try this out...

// ToDo:
//  - Quadratic Interpolation and Analytical Laplacian
//  - Implementation for the Momentum Transfer / Velocity Field...
//  - Optional Mixing Kernel / Time-Delay Kernel?
//  - Implementation of the Raw Erosion Kernel...
//  - Add the same solution but for a node-based velocity field...
//  Reimplement everything only on the python end as isolated kernels.

// Additional Potential Parameters for Solution Configuration:
//  -> epsilon
//  -> lambda_max

//! Uniform Sampling Monte-Carlo Estimator
silt::tensor solve_uniform (
  const silt::tensor_t<float> flow,     //!< Flow-Field Tensor
  const silt::tensor_t<float> source,   //!< Source Term Tensor
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source
  const silt::vec2 scale,               //!< Cell Scale
  const size_t count                    //!< Sample Count
);

//! Reservoir Sampling Implementation
// silt::tensor solve_reservoir();

//
//  Erosion Specific Source / Decay Kernels...
//

// silt::tensor suspend (
//   const silt::tensor_t<float> flow,     //!< Flow-Field Tensor
//   const silt::vec2 scale,               //!< Cell Scale
//   const float ks
// );

//
// Unified Erosion Kernel
//

void erode (
  silt::tensor_t<float> height,
  silt::tensor_t<silt::rng> rng,
  const silt::vec3 scale,
  const soil::param_t param
);

}


#endif