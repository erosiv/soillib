#pragma once

#include <soillib/soillib.hpp>

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! Bilinear Resample to a New Resolution
//! Resamples a 2D tensor (1, 2 or 3 vector components) from its current
//! resolution to shape_out, with clamped (edge-extended) boundaries.
silt::tensor_t<float> resize(const silt::tensor_t<float>& tensor, const silt::shape shape_out);

} // end of namespace soil
