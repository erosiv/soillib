#ifndef SOILLIB_LAYER_NORMAL
#define SOILLIB_LAYER_NORMAL

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <soillib/core/node.hpp>
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

template<typename T>
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

template<typename T>
glm::vec3 normal_impl(const soil::node& node, const soil::flat_t<2> index, glm::ivec2 p) {

//  auto _buffer = buffer.as<T>();

  // Gather 5x5 Region
  sample_t<T> px[5], py[5];
  for (size_t i = 0; i < 5; ++i) {

    const glm::ivec2 pos_x = p + glm::ivec2(-2 + i, 0);
    if (!index.oob(pos_x)) {
      px[i].oob = false;
      px[i].pos = pos_x;

      const size_t ind = index.flatten(pos_x);
      px[i].value = node.val<T>(ind);
    }

    const glm::ivec2 pos_y = p + glm::ivec2(0, -2 + i);
    if (!index.oob(pos_y)) {
      py[i].oob = false;
      py[i].pos = pos_y;

      const size_t ind = index.flatten(pos_y);
      py[i].value = node.val<T>(ind);
    }
  }

  // Compute Local Gradient
  const glm::vec2 g = gradient_detailed<T>(px, py);

  // Normal Vector from Gradient
  glm::vec3 n = glm::vec3(-g.x, -g.y, 1.0);
  if (length(n) > 0)
    n = normalize(n);
  return n;
}

} // namespace

// Surface Normal from Surface Gradient

//! Simple Transform Layer, which takes a 2D Layer
//! that returns some floating point type, and returns
//! the normal vector, either single sample or fully baked.
//!
struct normal {

  //! Single Sample Value
  static glm::vec3 operator()(soil::node node, soil::flat_t<2> index, const glm::ivec2 pos){
    // note: consider how we can make it strict-typed at this point already...
    // in other words: the node is generic, but internally it knows the type
    // of the nodes that are feeding it.
    return soil::select(node.type(), [node, index, pos]<typename T>() -> glm::vec3 {
      if constexpr (std::is_floating_point_v<T>)
        return normal_impl<T>(node, index, pos);
      else throw std::invalid_argument("invalid type for operation");
    });
  }

  static soil::node make_node(soil::index index, const soil::node& node) {

    if(index.type() != FLAT2)
      throw std::invalid_argument("can't extract a full noise buffer from a non-2D index");

    soil::select(node.type(), []<typename T>(){
      if constexpr (!std::is_floating_point_v<T>)
        std::invalid_argument("invalid type for operation");
    });

    auto index_t = index.as<soil::flat_t<2>>();

    using func_t = soil::map_t<glm::vec3>::func_t;
    using param_t = soil::map_t<glm::vec3>::param_t;

    const func_t func = [index_t, node](const param_t& in, const size_t i) -> glm::vec3 {
      soil::ivec2 position = index_t.unflatten(i);
      return soil::normal::operator()(node, index_t, position);
    };

    soil::map map = soil::map(func);
    return soil::node(map, {});

  }
};

} // end of namespace soil

#endif