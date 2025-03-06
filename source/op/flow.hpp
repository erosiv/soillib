#ifndef SOILLIB_LAYER_FLOW
#define SOILLIB_LAYER_FLOW

#include <random>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <soillib/soillib.hpp>
#include <soillib/util/error.hpp>

namespace soil {

//! \todo Make this generic! Constructing an operator like this should be much simpler.

namespace {

const double dirmap[8] = {
    7,
    8,
    1,
    2,
    3,
    4,
    5,
    6,
};

const std::vector<glm::ivec2> coords = {
    glm::ivec2{-1, 0},
    glm::ivec2{-1, 1},
    glm::ivec2{0, 1},
    glm::ivec2{1, 1},
    glm::ivec2{1, 0},
    glm::ivec2{1, -1},
    glm::ivec2{0, -1},
    glm::ivec2{-1, -1},
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

} // namespace

//! Compute the Indexed Flow Direction from a Height-Map
//! The flow directions are given by dirmap(7, 8, 1, 2, 3, 4, 5, 6),
//! corresponding to (N, NE, E, SE, S, SW, W, NW)
soil::buffer flow(const soil::buffer &buffer, const soil::index &index);

//! Compute the 2D Flow Direction from the Flow Index Buffer
soil::buffer direction(const soil::buffer &buffer, const soil::index &index);

//! Compute the Stochastic Accumulation from a 2D Flow Direction Buffer
soil::buffer accumulation(const soil::buffer &buffer, const soil::index &index, int iterations, size_t samples);

//! Compute the Weighted Stochastic Accumulation from a 2D Flow Direction Buffer
soil::buffer accumulation(const soil::buffer &direction, const soil::buffer &weights, const soil::index &index, int iterations, size_t samplesm, bool reservoir = true);

//! Compute the Exhaustive Accumulation from a 2D Flow Direction Buffer
soil::buffer accumulation_exhaustive(const soil::buffer &direction, const soil::index &index);

//! Compute the Exhaustive Accumulation from a 2D Flow Direction Buffer
soil::buffer accumulation_exhaustive(const soil::buffer &direction, const soil::index &index, const soil::buffer &weights);

//! Compute an Upstream Catchment Mask from a Flow Direction Buffer for a given Position
soil::buffer upstream(const soil::buffer &buffer, const soil::index &index, const glm::ivec2 target);

//! Compute the Upstream Distance from a Flow Direction Buffer for a given Position
soil::buffer distance(const soil::buffer &buffer, const soil::index &index, const glm::ivec2 target);

} // end of namespace soil

#endif