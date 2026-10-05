#pragma once

#include <soillib/soillib.hpp>

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! Hillslope creep rate: the divergence of a downhill sediment flux, as a
//! rate of surface change [m/time]. Integrate it outside:
//!   h += dt * creep(...) / scale.z
//! The sign is a change rate, not an erosion rate -- positive is accumulation,
//! so folding it into a net rate that gets SUBTRACTED (soil::spl_apply) wants
//! the opposite sign.
//!
//! Flux is taken per D8 link, from the higher cell to the lower one:
//!   q(S) = rate * S / (1 - (S/critslope)^2),   S = |dh| / link length
//! and the per-cell rate is sum_k w_k * q_k / d_k, signed by which way the
//! link runs. Links are weighted 2/3 axial and 1/3 diagonal so that a LINEAR
//! flux law reproduces the isotropic nine-point Laplacian exactly -- which is
//! the point of comparison for validating this against soil::laplacian.
//!
//! On the linear limit, and why the nonlinearity is the whole mechanism: a
//! per-link flux law that is antisymmetric in the pair is ALGEBRAICALLY the
//! ordinary Laplacian, because the downhill flux a cell receives from its
//! uphill neighbor is exactly the term a symmetric Laplacian already carries.
//! Restricting flux to downhill links therefore changes nothing on its own.
//! What makes transport one-way in effect is `critslope`: flux grows without
//! bound as a link approaches it, so a slope cannot steepen past it. That
//! yields convex crests at low slope and planar threshold slopes at the
//! critical angle, which is the morphology linear diffusion cannot produce.
//!
//! `critslope <= 0` disables the nonlinearity and gives plain linear
//! diffusion with diffusivity `rate`.
//!
//! The denominator is floored at 1 - 0.99^2, so the effective diffusivity is
//! bounded by ~50x `rate`. An explicit step is stable for
//! dt < min(scale.x, scale.y)^2 / (4 * 50 * rate); an exponentially-fitted
//! variant (as in soil::spl_apply_exp) is what lifts that.
//!
//! Boundaries are closed: a link that leaves the domain carries no flux, so
//! the interior conserves mass and material accumulates at the edge unless
//! something else removes it.
//!
//! \todo rate as a field rather than a scalar, for spatially varying
//!   erodibility; exponentially-fitted integration for large steps.
silt::tensor_t<float> creep (
  const silt::tensor_t<float>& height,  //!< Height Field              [1]
  const silt::vec3 scale,               //!< Cell Scale                [m]
  const float rate,                     //!< Creep Diffusivity         [m^2/time]
  const float critslope                 //!< Critical Slope, 0 = off   [1]
);

} // end of namespace soil
