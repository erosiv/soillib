#ifndef SOILLIB_OP_NORMAL
#define SOILLIB_OP_NORMAL

#include <soillib/soillib.hpp>
#include <soillib/core/shape.hpp>
#include <soillib/core/tensor.hpp>
#include <soillib/util/error.hpp>
#include <soillib/op/gather.hpp>

namespace soil {
namespace op {

//! Surface Normal from Surface Gradient
//! Uses a higher-order lerp function for gradient computation,
//! and then normalizes with z-component 1 to get surface normal.
//! This function operates on a tensor and returns a tensor.
//!
template<typename T>
soil::tensor normal(const soil::tensor_t<T>& tensor, const vec3 scale = vec3(1.0f)) {

  const soil::shape shape_in = tensor.shape();
  if (shape_in.dim != 2)
    throw std::invalid_argument("normal map can not be computed for non 2D-indexed buffers");  
  
  const soil::shape shape_out = soil::shape(shape_in[0], shape_in[1], 3);
  soil::tensor_t<T> output(shape_out);

  for(size_t i = 0; i < shape_in.elem; ++i) {
    
    const lerp5_t<T> lerp(tensor, shape_in.unflatten(i));
    const soil::vec2 g = lerp.grad(scale);
    const soil::vec3 n = glm::normalize(glm::vec3(-g.x, -g.y, 1.0));
    
    //! \todo add mechanism to write vec3s into float output...
    output[3*i + 0] = n.x;
    output[3*i + 1] = n.y;
    output[3*i + 2] = n.z;

  }

  return soil::tensor(std::move(output));

}

}
} // end of namespace soil

#endif