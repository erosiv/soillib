#pragma once

#include <soillib/soillib.hpp>

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! Separable "cone" field with prescribed per-axis gradients from an apex:
//!   U(x,y) = height - |dUdx * (x - apex.x) * scale.x|
//!                    - |dUdy * (y - apex.y) * scale.y|
//! A zero gradient on an axis makes the field flat (a ridge/line) along
//! that axis, e.g. dUdy = 0 gives a one-sided linear ramp in x only.
//! Meant as a small additive correction -- e.g. to give an 8-bit uplift
//! PNG a genuine per-pixel gradient where its quantization would
//! otherwise leave flat (zero-gradient) plateaus.
silt::tensor_t<float> cone (
  const silt::shape shape,
  const silt::vec2 apex,
  const float height,
  const float dUdx,
  const float dUdy,
  const silt::vec2 scale
);

} // end of namespace soil
