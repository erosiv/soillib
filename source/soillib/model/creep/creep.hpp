#pragma once

#include <soillib/soillib.hpp>

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! Hillslope creep rate: the divergence of a downhill sediment flux, as a rate
//! of surface change [m/time], positive for accumulation. Integrate outside as
//!   h += dt * creep(...) / scale.z
//!
//! Flux is taken per D8 link, from the higher cell of the pair to the lower:
//!   q(S) = rate * S / (1 - (S/critslope)^2),   S = |dh| / link length
//! and the cell rate is sum_k w_k * q_k / d_k, signed by which way the link
//! runs. Links are weighted 2/3 axial and 1/3 diagonal, so a linear flux law
//! reproduces the isotropic nine-point Laplacian exactly.
//!
//! `critslope <= 0` drops the nonlinearity, leaving linear diffusion with
//! diffusivity `rate`. Otherwise flux grows without bound as a link approaches
//! `critslope`, so a slope cannot steepen past it.
//!
//! The denominator is floored at 1 - 0.99^2, bounding the effective diffusivity
//! at ~50x `rate`, so an explicit step is stable for
//! dt < min(scale.x, scale.y)^2 / (4 * 50 * rate).
//!
//! Boundaries are closed: a link leaving the domain carries no flux.
//!
//! \todo rate as a field; exponentially-fitted integration for large steps.
silt::tensor_t<float> creep (
  const silt::tensor_t<float>& height,  //!< Height Field              [1]
  const silt::vec3 scale,               //!< Cell Scale                [m]
  const float rate,                     //!< Creep Diffusivity         [m^2/time]
  const float critslope                 //!< Critical Slope, 0 = off   [1]
);

} // end of namespace soil
