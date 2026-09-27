#pragma once

#include <soillib/soillib.hpp>

#include <silt/core/types.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! Surface normal of a height field.
//! Computes the surface gradient with the standard gradient kernel,
//! then transforms and normalizes it into a unit surface normal.
silt::tensor_t<float> normal(const silt::tensor_t<float>& tensor, const silt::vec3 scale = silt::vec3(1.0f));

} // end of namespace soil
