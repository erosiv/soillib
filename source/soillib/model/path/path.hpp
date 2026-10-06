#pragma once

#include <soillib/soillib.hpp>
#include <silt/core/tensor.hpp>
#include <soillib/model/path/erosion.hpp>

namespace soil {

//! Uniform-sampling Monte-Carlo estimator for the flux integral of a linear
//! conservation law. Particles are seeded uniformly, carry the source term of
//! their seed cell, and deposit it into every cell they cross. The estimate is
//! the flux in [X*m^D/time], not divided by a cell-crossing rate.
//!
//! The third dimension of the source tensor is the component count of the
//! transported quantity. Transport thus supports multi-channel linear laws.
//!
//! Particles start at rest and advance one cell per step. Particles are accelerated by
//! the gradient and damped by drag per unit distance. Flat fields produce no motion.
//! Particles terminate on leaving the domain, on a vanishing direction, on
//! attenuation below epsilon.
//!
silt::tensor solve_uniform (
  const silt::tensor_t<float> height,   //!< Height Field          [1]
  const silt::tensor_t<float> source,   //!< Source Term Tensor    [X/time/m^D]
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor     [1/m]
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source  [1]
  const silt::vec3 scale,               //!< Cell Scale            [m]
  const size_t count,                   //!< Sample Count          [1]
  const float drag                      //!< Inverse Drag Length   [1/m]
);

//! Reservoir Sampling Implementation
// silt::tensor solve_reservoir();

}
