#ifndef SOILLIB_MODEL
#define SOILLIB_MODEL

#include <soillib/core/index.hpp>
#include <soillib/core/matrix.hpp>
#include <soillib/core/node.hpp>

// #include <std::unordered_map>

// A model is effectively just a structure
// that lets us compose an index type with
// multiple different properties...

// Effectively this thing is a node container,
// which then somehow exposes the different
// nodes so they can be used by particles.

// The lookup should be fast enough...
// Do we want a fixed set of possible things?
// Because in that case we could just define
// an enumerator and use that to index properties...

namespace soil {

enum component {
  HEIGHT,   // Height Data
  MOMENTUM, // Momentum Data???
  DISCHARGE,
  MOMENTUM_TRACK,
  DISCHARGE_TRACK,
  RESISTANCE,
  MAXDIFF,
  SETTLING
};

struct model {

  using matrix_t = soil::matrix::singular;

  node &operator[](const soil::component comp) {
    return this->nodes[comp].value();
  }

  void set(const soil::component comp, const soil::node node) {
    this->nodes[comp] = node;
  }

  void add(const size_t index, const float value, const matrix_t matrix) {
    auto height = this->operator[](soil::component::HEIGHT);
    soil::select(height.type(), [self = this, index, &height, value]<typename S>() {
      auto cached = height.as<soil::cached>();
      auto buffer = cached.buffer.as<float>();
      buffer[index] += value;
    });
  }

  // private:
  soil::index index;
  std::array<std::optional<soil::node>, 8> nodes;
};

} // end of namespace soil

#endif