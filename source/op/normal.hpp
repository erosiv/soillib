#ifndef SOILLIB_LAYER_NORMAL
#define SOILLIB_LAYER_NORMAL

#include <soillib/soillib.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <soillib/util/error.hpp>

#include <soillib/op/gather.hpp>

namespace soil {

// Surface Normal from Surface Gradient

//! Simple Transform Layer, which takes a 2D Layer
//! that returns some floating point type, and returns
//! the normal vector, either single sample or fully baked.
//!
struct normal {

  template<std::floating_point T, typename I>
  static glm::vec3 operator()(soil::buffer_t<T> buffer_t, I index, const glm::ivec2 pos, const vec3 scale = vec3(1.0f)) {

    lerp5_t<T> lerp;
    lerp.gather(buffer_t, index, pos);
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
  static soil::buffer operator()(const soil::buffer &buffer, soil::index index, const vec3 scale = vec3(1.0f)) {

    static_assert(index_2D<soil::quad>, "test");

    if (index.dims() != 2)
      throw std::invalid_argument("normal map can not be computed for non 2D-indexed buffers");

    if (buffer.host() != soil::CPU)
      throw soil::error::mismatch_host(soil::CPU, buffer.host());

    return soil::select(index.type(), [&]<index_2D I>() -> soil::buffer {
      return soil::select(buffer.type(), [&]<std::floating_point T>() -> soil::buffer {
        auto index_t = index.as<I>();
        auto buffer_t = buffer.as<T>();

        soil::buffer_t<vec3> output(buffer.elem());
        for (auto [i, b] : output.iter()) {
          soil::ivec2 position = index_t.unflatten(i);
          *b = soil::normal::operator()(buffer_t, index_t, position, scale);
        }

        return soil::buffer(std::move(output));
      });
    });
  }

  static glm::vec3 operator()(soil::buffer buffer, soil::flat_t<2> index, const glm::ivec2 pos, const vec3 scale = vec3(1.0f)) {
    return soil::select(buffer.type(), [buffer, index, pos, scale]<std::floating_point T>() -> glm::vec3 {
      return soil::normal::operator()(buffer.as<T>(), index, pos, scale);
    });
  }

};

} // end of namespace soil

#endif