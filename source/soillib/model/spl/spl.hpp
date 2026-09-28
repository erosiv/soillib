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

} // end of namespace soil
