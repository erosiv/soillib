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
//! diffusivity `rate`. Otherwise the flux diverges at `critslope`, so a link
//! cannot steepen past it. At 0.99*critslope the law hands over to its own
//! tangent, bounding the diffusivity at 5000*rate while keeping dq/dS
//! continuous across the handover.
//!
//! Stability is governed by dq/dS, not by q/S: a perturbation changes a link
//! slope and the flux answers with the derivative, which near critslope is a
//! hundred times the secant. An explicit step is stable only for
//!   dt < 0.375 * min(scale.x, scale.y)^2 / (5000 * rate)
//! and the mode that goes first is the checkerboard. Use creep_relax below with
//! an exponentially-fitted update instead, which carries no step restriction.
//!
//! Boundaries are closed: a link leaving the domain carries no flux.
//!
//! \todo rate as a field.
silt::tensor_t<float> creep (
  const silt::tensor_t<float>& height,  //!< Height Field              [1]
  const silt::vec3 scale,               //!< Cell Scale                [m]
  const float rate,                     //!< Creep Diffusivity         [m^2/time]
  const float critslope                 //!< Critical Slope, 0 = off   [1]
);

//! Relaxation rate of soil::creep for an exponentially-fitted update:
//!   lambda = scale.z * sum_k w_k * dq/dS(S_k) / d_k^2
//! the diagonal of the linearized operator, so soil::spl_apply_exp caps the
//! effective step at 1/lambda and the cell relaxes onto its equilibrium slope
//! instead of overshooting it, at any step size. Differentiated against the
//! normalized height as soil::spl_relax is, so both take the same dt.
silt::tensor_t<float> creep_relax (
  const silt::tensor_t<float>& height,  //!< Height Field              [1]
  const silt::vec3 scale,               //!< Cell Scale                [m]
  const float rate,                     //!< Creep Diffusivity         [m^2/time]
  const float critslope                 //!< Critical Slope, 0 = off   [1]
);

} // end of namespace soil
