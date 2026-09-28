#pragma once

#include <soillib/soillib.hpp>

#include <silt/core/types.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! Surface normal from a precomputed 2-component gradient field
//! (normalize([-gx, -gy, 1])) -- pairs with any of soil::grad_steepest,
//! soil::grad_quadratic or soil::grad_centered (model/grad/grad.hpp), or
//! any other 2-channel gradient field.
silt::tensor_t<float> normal(const silt::tensor_t<float>& gradient);

//! Convenience: surface normal directly from a height field, via
//! soil::grad_centered and the normalize trick above.
silt::tensor_t<float> normal(const silt::tensor_t<float>& tensor, const silt::vec3 scale);

} // end of namespace soil
