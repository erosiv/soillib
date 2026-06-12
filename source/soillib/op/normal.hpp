#ifndef SOILLIB_OP_NORMAL
#define SOILLIB_OP_NORMAL

#include <soillib/soillib.hpp>
#include <silt/core/shape.hpp>
#include <silt/core/tensor.hpp>
#include <silt/core/error.hpp>
#include <silt/op/gather.hpp>

namespace soil {
namespace op {

//! Surface Normal from Surface Gradient
//! Uses a higher-order lerp function for gradient computation,
//! and then normalizes with z-component 1 to get surface normal.
//! This function operates on a tensor and returns a tensor.
//!
template<typename T>
silt::tensor normal(const silt::tensor_t<T>& tensor, const vec3 scale = vec3(1.0f)) {

  const silt::shape shape_in = tensor.shape();
  if (shape_in.dim() != 2)
    throw std::invalid_argument("normal map can not be computed for non 2D-indexed buffers");  
  
  const silt::shape shape_out = silt::shape(shape_in[0], shape_in[1], 3);
  silt::tensor_t<T> output(shape_out);
  silt::view_t<silt::vec3> test = output.view<silt::vec3>();

  for(size_t i = 0; i < shape_in.elem(); ++i) {
    
    const lerp5_t<T> lerp(tensor, shape_in.unflatten(i));
    const silt::vec2 g = lerp.grad(scale);
    test[i] = glm::normalize(glm::vec3(-g.x, -g.y, 1.0));

  }

  return silt::tensor(std::move(output));

}

}
} // end of namespace soil

#endif