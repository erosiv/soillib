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

}

//! Surface Flow Operator
//!
//! Operates on the principle of steepest direction.
//!
struct flow {

  flow(soil::index index, const soil::node &node): index{index} {
    auto cached = node.as<soil::cached>();
    this->buffer = soil::buffer(cached.as<double>().buffer);
  }

  //! Bake a whole buffer!
  //! Note: we make sure that the indexing structure of the buffer is respected.
  soil::buffer full() const {

    return soil::select(index.type(), [self = this]<typename T>() -> soil::buffer {
      if constexpr (std::same_as<typename T::vec_t, soil::ivec2>) {

        auto index = self->index.as<T>();
        auto in = self->buffer.as<double>();
        auto out = buffer_t<int>{index.elem()};

        for (const auto &pos : index.iter()) {
        
          const size_t i = index.flatten(pos);
          
          double diffmax = 0.0f;
          double hvalue = in[i];
          int value = -2;   // default also for nan
          bool pit = true;
          bool has_flow = false;

          for(size_t k = 0; k < 8; ++k){

            const glm::ivec2 coord = coords[k];
            const glm::ivec2 npos = pos + coord;
            if(!index.oob(npos)){
              
              const size_t n = index.flatten(npos);
              const double nvalue = in[n];
              const double dist = sqrt(coord[0]*coord[0] + coord[1]*coord[1]);
              const double ndiff = (hvalue - nvalue)/dist;
              
              if(ndiff > diffmax){
                value = k;
                diffmax = ndiff;
              }

              if(ndiff > 0.0)            
                has_flow = true;
            
              if(ndiff >= 0.0)
                pit = false;

            }

          }

          if(pit) value = -2;
          if(!has_flow && !pit) value = -1;
          
          if(value >= 0)
            out[i] = dirmap[value];
          else out[i] = value;
  
        }

        return std::move(soil::buffer(std::move(out)));

      } else {
        throw std::invalid_argument("can't extract a full flow buffer from a non-2D index");
      }
    });
  }

private:
  soil::index index;
  soil::buffer buffer;
};

struct direction {

  direction(soil::index index, const soil::node &node): index{index} {
    auto cached = node.as<soil::cached>();
    this->buffer = soil::buffer(cached.as<int>().buffer);
  }

  //! Bake a whole buffer!
  //! Note: we make sure that the indexing structure of the buffer is respected.
  soil::buffer full() const {

    return soil::select(index.type(), [self = this]<typename T>() -> soil::buffer {
      if constexpr (std::same_as<typename T::vec_t, soil::ivec2>) {

        auto index = self->index.as<T>();
        const size_t elem = index.elem();
        auto in = self->buffer.as<int>();
        auto out = buffer_t<ivec2>{elem};

        for(size_t i = 0; i < elem; ++i){
          glm::ivec2 val(0, 0);
          for(size_t k = 0; k < 8; ++k){
            if(in[i] == dirmap[k])
              val = coords[k];
          }
          out[i] = val;
        }

        return std::move(soil::buffer(std::move(out)));

      } else {
        throw std::invalid_argument("can't extract a full flow buffer from a non-2D index");
      }
    });
  }

private:
  soil::index index;
  soil::buffer buffer;
};

struct accumulation {

  accumulation(soil::index index, const soil::node &node): index{index} {
    auto cached = node.as<soil::cached>();
    this->buffer = soil::buffer(cached.as<int>().buffer);
  }

  //! Bake a whole buffer!
  //! Note: we make sure that the indexing structure of the buffer is respected.
  soil::buffer full() const {

    return soil::select(index.type(), [self = this]<typename T>() -> soil::buffer {
      if constexpr (std::same_as<typename T::vec_t, soil::ivec2>) {

        auto index = self->index.as<T>();
        const size_t elem = index.elem();
        auto in = self->buffer.as<ivec2>();
        auto out = buffer_t<double>{elem};

        for(size_t i = 0; i < elem; ++i)
          out[i] = 0.0;

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist_x(0, index[0]-1);
        std::uniform_int_distribution<std::mt19937::result_type> dist_y(0, index[1]-1);

        const size_t iterations = 2048;
        const size_t samples = 1024;
        const size_t steps = 3072;

        for(size_t i = 0; i < iterations; ++i){

          std::cout<<i<<std::endl;

          const double P = double(i * samples) / double(elem);

          for(size_t n = 0; n < samples; ++n){

            ivec2 pos{dist_x(rng), dist_y(rng)};
            size_t ind = index.flatten(pos);

            for(size_t s = 0; s < steps; ++s){

              const ivec2 dir = in[ind];
              if(dir[0] == 0 && dir[1] == 0)
                break;
            
              pos += dir;

              if(index.oob(pos))
                break;

              ind = index.flatten(pos);
              out[ind] += 1;

            }

          }

        }

        const double P = double(elem)/double(iterations*(samples));
        for(size_t i = 0; i < elem; i++){
          out[i] = 1.0 + P * out[i];
        }

        return std::move(soil::buffer(std::move(out)));

      } else {
        throw std::invalid_argument("can't extract a full flow buffer from a non-2D index");
      }
    });
  }

private:
  soil::index index;
  soil::buffer buffer;
};

} // end of namespace soil

#endif