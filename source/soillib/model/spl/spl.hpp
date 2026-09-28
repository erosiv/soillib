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

} // end of namespace soil
