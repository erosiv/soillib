#ifndef SOILLIB_LAYER_NORMAL
#define SOILLIB_LAYER_NORMAL

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <soillib/soillib.hpp>
#include <soillib/util/error.hpp>

namespace soil {

namespace {

template<typename T>
struct sample_t {
  glm::ivec2 pos;
  T value;
  bool oob = true;
};

template<typename T, typename I>
void gather(const soil::buffer_t<T> &buffer_t, const I index, glm::ivec2 p, sample_t<T> px[5], sample_t<T> py[5]) {
  for (size_t i = 0; i < 5; ++i) {

    const glm::ivec2 pos_x = p + glm::ivec2(-2 + i, 0);
    if (!index.oob(pos_x)) {
      px[i].oob = false;
      px[i].pos = pos_x;

      const size_t ind = index.flatten(pos_x);
      px[i].value = buffer_t[ind];
    }

    const glm::ivec2 pos_y = p + glm::ivec2(0, -2 + i);
    if (!index.oob(pos_y)) {
      py[i].oob = false;
      py[i].pos = pos_y;

      const size_t ind = index.flatten(pos_y);
      py[i].value = buffer_t[ind];
    }
  }
}

template<std::floating_point T>
glm::vec2 gradient_detailed(sample_t<T> px[5], sample_t<T> py[5]) {

  glm::vec2 g = glm::vec2(0, 0);

  // X-Element
  if (!px[0].oob && !px[4].oob)
    g.x = (1.0f * px[0].value - 8.0f * px[1].value + 8.0f * px[3].value - 1.0f * px[4].value) / 12.0f;

  else if (!px[0].oob && !px[3].oob)
    g.x = (1.0f * px[0].value - 6.0f * px[1].value + 3.0f * px[2].value + 2.0f * px[3].value) / 6.0f;

  else if (!px[0].oob && !px[2].oob)
    g.x = (1.0f * px[0].value - 4.0f * px[1].value + 3.0f * px[2].value) / 2.0f;

  else if (!px[1].oob && !px[4].oob)
    g.x = (-2.0f * px[1].value - 3.0f * px[2].value + 6.0f * px[3].value - 1.0f * px[4].value) / 6.0f;

  else if (!px[2].oob && !px[4].oob)
    g.x = (-3.0f * px[2].value + 4.0f * px[3].value - 1.0f * px[4].value) / 2.0f;

  else if (!px[1].oob && !px[3].oob)
    g.x = (-1.0f * px[1].value + 1.0f * px[3].value) / 2.0f;

  else if (!px[2].oob && !px[3].oob)
    g.x = (-1.0f * px[2].value + 1.0f * px[3].value) / 1.0f;

  else if (!px[1].oob && !px[2].oob)
    g.x = (-1.0f * px[1].value + 1.0f * px[2].value) / 1.0f;

  // Y-Element

  if (!py[0].oob && !py[4].oob)
    g.y = (1.0f * py[0].value - 8.0f * py[1].value + 8.0f * py[3].value - 1.0f * py[4].value) / 12.0f;

  else if (!py[0].oob && !py[3].oob)
    g.y = (1.0f * py[0].value - 6.0f * py[1].value + 3.0f * py[2].value + 2.0f * py[3].value) / 6.0f;

  else if (!py[0].oob && !py[2].oob)
    g.y = (1.0f * py[0].value - 4.0f * py[1].value + 3.0f * py[2].value) / 2.0f;

  else if (!py[1].oob && !py[4].oob)
    g.y = (-2.0f * py[1].value - 3.0f * py[2].value + 6.0f * py[3].value - 1.0f * py[4].value) / 6.0f;

  else if (!py[2].oob && !py[4].oob)
    g.y = (-3.0f * py[2].value + 4.0f * py[3].value - 1.0f * py[4].value) / 2.0f;

  else if (!py[1].oob && !py[3].oob)
    g.y = (-1.0f * py[1].value + 1.0f * py[3].value) / 2.0f;

  else if (!py[2].oob && !py[3].oob)
    g.y = (-1.0f * py[2].value + 1.0f * py[3].value) / 1.0f;

  else if (!py[1].oob && !py[2].oob)
    g.y = (-1.0f * py[1].value + 1.0f * py[2].value) / 1.0f;

  return g;
}

} // namespace

// Surface Normal from Surface Gradient

//! Simple Transform Layer, which takes a 2D Layer
//! that returns some floating point type, and returns
//! the normal vector, either single sample or fully baked.
//!
struct normal {

  template<std::floating_point T, typename I>
  static glm::vec3 operator()(soil::buffer_t<T> buffer_t, I index, const glm::ivec2 pos) {

    sample_t<T> px[5], py[5];
    gather<T, I>(buffer_t, index, pos, px, py);
    const glm::vec2 g = gradient_detailed<T>(px, py);
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
  static soil::buffer operator()(const soil::buffer &buffer, soil::index index) {

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
          *b = soil::normal::operator()(buffer_t, index_t, position);
        }

        return soil::buffer(std::move(output));
      });
    });
  }

  static glm::vec3 operator()(soil::buffer buffer, soil::flat_t<2> index, const glm::ivec2 pos) {
    return soil::select(buffer.type(), [buffer, index, pos]<std::floating_point T>() -> glm::vec3 {
      return soil::normal::operator()(buffer.as<T>(), index, pos);
    });
  }

};

} // end of namespace soil

#endif