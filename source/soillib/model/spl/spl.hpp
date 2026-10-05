#pragma once

#include <soillib/soillib.hpp>

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! Local incision rate under a pure stream-power law, written into `rate`:
//!   E = K * Q^m * |S|^n
//! Detachment-limited, no deposition. Q is a discharge field (e.g. from
//! soil::accumulate), S a downstream slope field (e.g. from soil::slope) --
//! sign-agnostic, so either soil::slope's convention or an externally
//! computed slope magnitude works.
void spl_rate (
  silt::tensor_t<float> rate,
  const silt::tensor_t<float> discharge,
  const silt::tensor_t<float> slope,
  const float K,
  const float m,
  const float n
);

//! Apply an erosion rate to a height field in place: h -= dt * rate
void spl_apply (
  silt::tensor_t<float> height,
  const silt::tensor_t<float> rate,
  const float dt
);

//! Linearized relaxation rate for the exponentially-fitted update below,
//! written into `relax`:
//!   lambda = dE/dh = n * K * Q^m * |S|^(n-1) * dS/dh   [1/time]
//! dS/dh is the sensitivity of an upwind slope to the local height, i.e.
//! 1/dx with the vertical scale folded in (scale.z/min(scale.x, scale.y)):
//! the steeper axis is taken, so the fit errs toward more damping.
//! For n = 1 this collapses to lambda = K * Q^m * dS/dh, independent of S.
//! Only meaningful for an upwind slope (grad_steepest/grad_quadratic) --
//! a centered stencil does not involve the local height at all, so its
//! dS/dh is zero and no fitting is possible.
void spl_relax (
  silt::tensor_t<float> relax,
  const silt::tensor_t<float> discharge,
  const silt::tensor_t<float> slope,
  const float K,
  const float m,
  const float n,
  const silt::vec3 scale
);

//! Exponentially-fitted height update, in place:
//!   h -= dt * phi(lambda*dt) * rate,   phi(z) = (1 - exp(-z))/z
//! Treating the receiver height as frozen over the step, the height obeys
//! dh/dt = -lambda*(h - h_eq), whose exact solution is the above. phi(0)=1
//! recovers forward Euler, and phi ~ 1/z for large z caps the effective
//! step at 1/lambda, so the cell relaxes onto h_eq instead of overshooting
//! it -- stable for any dt, with the steady state (where the net rate
//! vanishes) preserved exactly.
//! `rate` must carry every source term, uplift included, or h_eq -- and so
//! the equilibrium slope -- is wrong at large dt.
void spl_apply_exp (
  silt::tensor_t<float> height,
  const silt::tensor_t<float> rate,
  const silt::tensor_t<float> relax,
  const float dt
);

//! Cap a deposition rate at what a cell can aggrade in one step without
//! filling toward its donors, in place:
//!   D <- min(D, (fraction * slope * length + floor) / dt)
//! The headroom a cell has before deposition reverses its flow scales with its
//! local relief, slope * length; `floor` is the rise a flat is still allowed,
//! without which a zero slope admits no deposition at all.
//!
//! Pointwise and monotone, so unlike soil::spill_limit it couples no
//! neighbours and cannot imprint a spatial pattern of its own.
void spl_deposition_limit (
  silt::tensor_t<float> deposition,
  const silt::tensor_t<float> slope,
  const float fraction,
  const float length,
  const float floor,
  const float dt
);

//! Cap aggradation at the level of the cells that drained into it, as a
//! new tensor. `before` is the height field as it stood at the start of the
//! step and fixes which D4 neighbors are donors; `height` is the updated
//! field to clamp. A cell rising above its donors reverses their flow and
//! strands them as pits, which single-receiver routing can not resolve --
//! the backstop for deposition in flats, where the discharge is too small
//! for spl_relax's damping to bite. Non-conservative: the excess is
//! discarded rather than passed downstream.
silt::tensor_t<float> spill_limit (
  const silt::tensor_t<float>& height,
  const silt::tensor_t<float>& before,
  const float epsilon
);

//! Stream-power-parameterized diffusivity, written into `diffusivity`:
//!   D = D0 + k_D * Q^m * |S|^n
//! Same form and inputs as spl_rate, plus a base (uniform) diffusivity D0.
//! Intended to drive a per-cell soil::gaussian_blur sigma field, as a fast
//! approximation of deposition -- more discharge/slope locally diffuses
//! (redistributes) more material, on top of a uniform background rate.
void spl_diffusivity (
  silt::tensor_t<float> diffusivity,
  const silt::tensor_t<float> discharge,
  const silt::tensor_t<float> slope,
  const float D0,
  const float k_D,
  const float m,
  const float n
);

//! Decay rate of a transported load [1/m], written into `decay`: the
//! reciprocal of the length over which concentration-based deposition relaxes
//! the load onto the local capacity,
//!   lambda = Qw / (k_S * width)
//! which is the attenuation soil::solve_uniform applies per unit path length.
//! Discharge is floored at `minimum` so a dry cell gives a finite, large rate.
void spl_decay (
  silt::tensor_t<float> decay,
  const silt::tensor_t<float> discharge,
  const float k_S,
  const float width,
  const float minimum
);

//! Concentration-based deposition rate, written into `deposition`:
//!   D = k_S * Qs / Qw
//! Qs is a sediment-discharge field (e.g. soil::accumulate of a spl_rate
//! erosion-rate source along the same flow graph used for Qw), Qw the
//! water-discharge field it's normalized against. Qw is floored away from
//! zero so dry cells don't divide by it.
void spl_deposition (
  silt::tensor_t<float> deposition,
  const silt::tensor_t<float> Qs,
  const silt::tensor_t<float> Qw,
  const float k_S
);

} // end of namespace soil
