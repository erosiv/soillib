#pragma once

#include <soillib/soillib.hpp>

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! Isotropic Gaussian Blur (uniform sigma)
silt::tensor_t<float> gaussian_blur(silt::tensor_t<float> tensor, const float sigma);

//! Isotropic Gaussian Blur w/ Spatially-Varying Sigma
silt::tensor_t<float> gaussian_blur(silt::tensor_t<float> tensor, const silt::tensor_t<float> sigma);

}
