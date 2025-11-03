#ifndef SOILLIB_MODEL_SCULPT
#define SOILLIB_MODEL_SCULPT

#include <silt/core/types.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>

namespace soil {

void masked_set(silt::tensor_t<float> tensor, const float value, const silt::vec2 center, const float rad);

}

#endif