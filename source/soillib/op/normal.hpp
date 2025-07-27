#ifndef SOILLIB_LAYER_NORMAL
#define SOILLIB_LAYER_NORMAL

#include <soillib/soillib.hpp>
#include <soillib/core/shape.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/util/error.hpp>
#include <soillib/op/gather.hpp>

namespace soil {

// Surface Normal from Surface Gradient

//! Simple Transform Layer, which takes a 2D Layer
//! that returns some floating point type, and returns
//! the normal vector, either single sample or fully baked.
//!
struct normal {

  template<std::floating_point T>
  static glm::vec3 operator()(soil::buffer_t<T> buffer_t, const soil::shape& shape, const glm::ivec2 pos, const vec3 scale = vec3(1.0f)) {

    lerp5_t<T> lerp;
    lerp.gather(buffer_t, shape, pos);
    const glm::vec2 g = lerp.grad(scale);
    glm::vec3 n = glm::vec3(-g.x, -g.y, 1.0);
    if (length(n) > 0) {
      n = normalize(n);
    }
    return n;
  }

  //
  // Full Sampling Functions
  //

  // Direct Execution
  //!\todo replace this with a tensor type so the shape conversion is automatic.
  static soil::buffer operator()(const soil::buffer &buffer, const soil::shape& shape, const vec3 scale = vec3(1.0f)) {

    if (shape.dim != 2)
      throw std::invalid_argument("normal map can not be computed for non 2D-indexed buffers");

    if (buffer.host() != soil::CPU)
      throw soil::error::mismatch_host(soil::CPU, buffer.host());

    return soil::select(buffer.type(), [&]<std::floating_point T>() -> soil::buffer {

      auto buffer_t = buffer.as<T>();
      soil::buffer_t<T> output(3*buffer.elem());

      for(size_t i = 0; i < shape.elem; ++i){

        soil::ivec2 position = shape.unflatten(i);
        vec3 value = soil::normal::operator()(buffer_t, shape, position, scale);
        output[3*i + 0] = value.x;
        output[3*i + 1] = value.y;
        output[3*i + 2] = value.z;

      }

      return soil::buffer(std::move(output));
      
    });
  }

};

} // end of namespace soil

#endif