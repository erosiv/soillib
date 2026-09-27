#pragma once

#include <soillib/soillib.hpp>

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! 2D Tensor Gradient (Godunov Min-Slope)
silt::tensor_t<float> gradient(const silt::tensor_t<float>& tensor, const silt::vec2 scale);

//! 2D Tensor Laplacian
silt::tensor_t<float> laplacian(const silt::tensor_t<float>& tensor, const silt::vec2 scale);

//! Safe Negative Slope Copmutation? Zero in pits?
silt::tensor_t<float> negslope(const silt::tensor_t<float>& tensor, const silt::vec2 scale);

}
