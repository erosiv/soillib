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
//! transported quantity, and may be 1, 2 or 3.
//!
//! MOTION. A particle advances one cell per step along a direction that relaxes
//! onto the local steepest descent over an arc-length scale of 1/drag metres:
//!
//!   d <- d_eq + (d - d_eq) * exp(-drag * ds),   d_eq = -grad(h)
//!
//! `drag` is an inverse length. The direction memory is 1/(drag*|scale.xy|)
//! cells, and that product is the resolution-invariant group, so halving the
//! cell size wants drag doubled; above a product of about 1 a step is plain
//! steepest descent. The magnitude of d does not affect the trajectory.
//!
//! Particles start at rest, so the first direction is steepest descent, and a
//! flat field produces no motion.
//!
//! `decay` attenuates the carried quantity per unit path length, so 1/decay is
//! a mixing length in metres.
//!
//! Particles terminate on leaving the domain, on a vanishing direction, on
//! attenuation below epsilon, on a Manhattan step bound, and on coasting one
//! coasting length over ground with no descent: the step is one cell whatever
//! the direction's magnitude, so a particle crossing a flat would otherwise
//! carry its inherited direction straight out of it.
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
