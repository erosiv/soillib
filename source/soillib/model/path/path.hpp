#pragma once

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

// ToDo:
//  - Quadratic Interpolation and Analytical Laplacian
//  - Optional Mixing Kernel / Time-Delay Kernel?
//  - Implementation of the Raw Erosion Kernel...
//  Reimplement everything only on the python end as isolated kernels.

//! Uniform Sampling Monte-Carlo Estimator for the flux integral of a linear
//! conservation law, as an alternative to graph-based accumulation.
//!
//! Particles are seeded uniformly, carry the source term of their seed cell,
//! and deposit it into every cell they cross. The estimate is the flux itself
//! in [X*m^D/time] -- an actual discharge, NOT divided by a cell-crossing
//! rate, so the coefficients of a law consuming it (e.g. K in soil::spl_rate)
//! stay dimensional and transfer across resolutions and domain sizes.
//!
//! The third dimension of the source tensor is the dimensionality of the
//! transported quantity: 1, 2 or 3 components, so an RGB field rides along
//! with the same estimator that carries water.
//!
//! MOTION. The velocity field is not prescribed. A particle is accelerated by
//! the local potential gradient against a linear drag, so its terminal
//! velocity is v_eq = -(gravity/drag) * grad(h), and over a step of duration
//! dt it relaxes toward that exactly:
//!   v <- v_eq + (v - v_eq) * exp(-drag * dt)
//! This is the quasi-static transport assumption: the flow is taken to be
//! equilibrated, which removes the time axis and makes the voxel-crossing
//! step a valid dynamic timestep. The integration is exact for a gradient
//! held constant over the step, so it is unconditionally stable: there is no
//! step-size constraint, and a large step simply lands on v_eq.
//!
//! `drag` is therefore the one knob that decides how far a particle coasts
//! against an adverse gradient, which is what carries flow across a shallow
//! pit instead of trapping it the way steepest descent does.
//!
//! SETTING IT. The velocity relaxes by exp(-drag * dt) per cell crossing, and
//! at terminal velocity dt = L/v_eq, so the memory of the motion is governed
//! by one dimensionless number -- the coasting distance in cells:
//!
//!   St = gravity * S / (drag^2 * L)       L = |scale.xy|, S a typical slope
//!
//! St >> 1 coasts, St << 1 relaxes onto the terminal velocity within a single
//! cell, which IS steepest descent -- the estimator then reproduces graph-based
//! accumulation and the pits come back. The admissible range is therefore
//!
//!   drag << sqrt(gravity * S / L)
//!
//! which for 8 m cells on a 0.05 slope is drag << 0.2: a drag of 1 gives
//! St = 0.04 and no inertia whatsoever. There is no safe default, which is why
//! this parameter has none.
//!
//! Inverting the energy criterion -- a particle crosses a lip of height dh when
//! its kinetic energy exceeds gravity * dh -- gives the depression depth the
//! flow will ignore, measured to be ~1.5 * gravity * S^2 / drag^2 (above the
//! bare criterion because the background slope keeps accelerating the particle
//! as it climbs). So to treat depressions shallower than dh as noise:
//!
//!   drag ~ S * sqrt(1.5 * gravity / dh)
//!
//! Choosing it this way is a modelling statement about what counts as noise.
//! Too small and no closed basin can ever form, which costs endorheic basins.
//!
//! `gravity` is an acceleration in the MODEL's own time unit, not 9.81 unless
//! that unit is the second -- a model stepping in years would want ~1e16. Only
//! gravity/drag (the velocity scale) and drag (the memory) are independent, so
//! the pair is really those two quantities and gravity alone means nothing.
//! Neither parameter has a default for that reason.
//!
//! A perfectly flat field has a zero terminal velocity everywhere, so no
//! particle moves and the estimate collapses to the local source term. Some
//! initial relief -- a regional tilt, or noise -- is load bearing.
//!
//! Particles terminate on leaving the domain, on a vanishing velocity, on
//! attenuation below epsilon, and on a Manhattan step bound, which is also
//! what caps a particle orbiting a flat.
silt::tensor solve_uniform (
  const silt::tensor_t<float> height,   //!< Height Field              [1]
  const silt::tensor_t<float> source,   //!< Source Term Tensor        [X/time/m^D]
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor         [1/time]
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source      [1]
  const silt::vec3 scale,               //!< Cell Scale                [m]
  const size_t count,                   //!< Sample Count              [1]
  const float gravity,                  //!< Gravitational Acceleration [m/time^2]
  const float drag                      //!< Linear Drag Rate          [1/time]
);

//! Reservoir Sampling Implementation
// silt::tensor solve_reservoir();

}
