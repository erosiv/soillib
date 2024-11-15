#ifndef SOILLIB_LAYER_FLOW
#define SOILLIB_LAYER_FLOW

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <soillib/core/node.hpp>
#include <soillib/soillib.hpp>
#include <soillib/util/error.hpp>
#include <random>

namespace soil {

//! \todo Make this generic! Constructing an operator like this should be much simpler.

namespace {

const double dirmap[8] = {
  7, 8, 1, 2, 3, 4, 5, 6,
};

const std::vector<glm::ivec2> coords = {
  glm::ivec2{-1, 0},
  glm::ivec2{-1, 1},
  glm::ivec2{ 0, 1},
  glm::ivec2{ 1, 1},
  glm::ivec2{ 1, 0},
  glm::ivec2{ 1,-1},
  glm::ivec2{ 0,-1},
  glm::ivec2{-1,-1},
};

const std::vector<double> dist = {
  1.0,
  sqrt(2.0),
  1.0,
  sqrt(2.0),
  1.0,
  sqrt(2.0),
  1.0,
  sqrt(2.0)
};

}

soil::buffer flow(const soil::buffer& buffer, const soil::index& index);
soil::buffer direction(const soil::buffer& buffer, const soil::index& index);
soil::buffer accumulation(const soil::buffer& buffer, const soil::index& index, int iterations, int samples, int steps);
soil::buffer upstream(const soil::buffer& buffer, const soil::index& index, const glm::ivec2 target);
soil::buffer distance(const soil::buffer& buffer, const soil::index& index, const glm::ivec2 target);

} // end of namespace soil

#endif