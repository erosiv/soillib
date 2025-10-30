#ifndef SOILLIB_MODEL_GRAD
#define SOILLIB_MODEL_GRAD

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

//! 2D Tensor Gradient (Godunov Min-Slope)
silt::tensor_t<float> gradient(const silt::tensor_t<float>& tensor, const silt::vec2 scale);

}

#endif