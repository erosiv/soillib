#ifndef SOILLIB_MODEL_FILTER
#define SOILLIB_MODEL_FILTER

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! Isotropic Gaussian Blur
silt::tensor_t<float> gaussian_blur(silt::tensor_t<float> tensor, const float sigma);

//! 2D Tensor Gradient (Godunov Min-Slope)
silt::tensor_t<float> gradient(const silt::tensor_t<float>& tensor, const silt::vec2 scale);

}

#endif