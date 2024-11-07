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

//! Surface Flow Operator
//!
//! Operates on the principle of steepest direction.
//!
struct flow {

  flow(soil::index index, const soil::node &node){
    
    soil::select(index.type(), [self = this, index]<typename T>() {
      if constexpr (std::same_as<typename T::vec_t, soil::ivec2>) {
        self->index = index.as<flat_t<2>>();
      } else {
        throw std::invalid_argument("can't extract a full flow buffer from a non-2D index");
      }
    });

    auto cached = node.as<soil::cached>();
    this->buffer = soil::buffer(cached.as<double>().buffer);
  }

  //! Bake a whole buffer!
  //! Note: we make sure that the indexing structure of the buffer is respected.
  soil::buffer full() const;

private:
  soil::flat_t<2> index;
  soil::buffer buffer;
};

struct direction {

  direction(soil::index index, const soil::node &node){

    soil::select(index.type(), [self = this, index]<typename T>() {
      if constexpr (std::same_as<typename T::vec_t, soil::ivec2>) {
        self->index = index.as<flat_t<2>>();
      } else {
        throw std::invalid_argument("can't extract a full flow buffer from a non-2D index");
      }
    });

    auto cached = node.as<soil::cached>();
    this->buffer = soil::buffer(cached.as<int>().buffer);
  }

  //! Bake a whole buffer!
  //! Note: we make sure that the indexing structure of the buffer is respected.
  soil::buffer full() const;

private:
  soil::flat_t<2> index;
  soil::buffer buffer;
};

struct accumulation {

  accumulation(soil::index index, const soil::node &node) {

    soil::select(index.type(), [self = this, index]<typename T>() {
      if constexpr (std::same_as<typename T::vec_t, soil::ivec2>) {
        self->index = index.as<flat_t<2>>();
      } else {
        throw std::invalid_argument("can't extract a full flow buffer from a non-2D index");
      }
    });

    auto cached = node.as<soil::cached>();
    this->buffer = soil::buffer(cached.as<int>().buffer);
  }

  size_t iterations = 128;
  size_t samples = 1024;
  size_t steps = 3072;

  //! Bake a whole buffer!
  //! Note: we make sure that the indexing structure of the buffer is respected.
  soil::buffer full() const;

private:
  soil::flat_t<2> index;
  soil::buffer buffer;
};

} // end of namespace soil

#endif