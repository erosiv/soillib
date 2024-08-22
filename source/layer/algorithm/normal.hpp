#ifndef SOILLIB_LAYER_NORMAL
#define SOILLIB_LAYER_NORMAL

#include <soillib/soillib.hpp>
#include <soillib/core/index.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/layer/layer.hpp>

namespace soil {

namespace {

template<typename T>
struct sample_t {
  glm::ivec2 pos;
  T value;
  bool oob = true;
};

template<typename T>
glm::vec2 gradient_detailed(sample_t<T> px[5], sample_t<T> py[5]){

  glm::vec2 g = glm::vec2(0, 0);

  // X-Element  
  if(!px[0].oob && !px[4].oob)
    g.x = (1.0f*px[0].value - 8.0f*px[1].value + 8.0f*px[3].value - 1.0f*px[4].value)/12.0f;

  else if(!px[0].oob && !px[3].oob)
    g.x = (1.0f*px[0].value - 6.0f*px[1].value + 3.0f*px[2].value + 2.0f*px[3].value)/6.0f;

  else if(!px[0].oob && !px[2].oob)
    g.x = (1.0f*px[0].value - 4.0f*px[1].value + 3.0f*px[2].value)/2.0f;

  else if(!px[1].oob && !px[4].oob)
    g.x = (-2.0f*px[1].value - 3.0f*px[2].value + 6.0f*px[3].value - 1.0f*px[4].value)/6.0f;

  else if(!px[2].oob && !px[4].oob)
    g.x = (-3.0f*px[2].value + 4.0f*px[3].value - 1.0f*px[4].value)/2.0f;

  else if(!px[1].oob && !px[3].oob)
    g.x = (-1.0f*px[1].value + 1.0f*px[3].value)/2.0f;

  else if(!px[2].oob && !px[3].oob)
    g.x = (-1.0f*px[2].value + 1.0f*px[3].value)/1.0f;

  else if(!px[1].oob && !px[2].oob)
    g.x = (-1.0f*px[1].value + 1.0f*px[2].value)/1.0f;

  // Y-Element

  if(!py[0].oob && !py[4].oob)
    g.y = (1.0f*py[0].value - 8.0f*py[1].value + 8.0f*py[3].value - 1.0f*py[4].value)/12.0f;

  else if(!py[0].oob && !py[3].oob)
    g.y = (1.0f*py[0].value - 6.0f*py[1].value + 3.0f*py[2].value + 2.0f*py[3].value)/6.0f;

  else if(!py[0].oob && !py[2].oob)
    g.y = (1.0f*py[0].value - 4.0f*py[1].value + 3.0f*py[2].value)/2.0f;

  else if(!py[1].oob && !py[4].oob)
    g.y = (-2.0f*py[1].value - 3.0f*py[2].value + 6.0f*py[3].value - 1.0f*py[4].value)/6.0f;

  else if(!py[2].oob && !py[4].oob)
    g.y = (-3.0f*py[2].value + 4.0f*py[3].value - 1.0f*py[4].value)/2.0f;

  else if(!py[1].oob && !py[3].oob)
    g.y = (-1.0f*py[1].value + 1.0f*py[3].value)/2.0f;

  else if(!py[2].oob && !py[3].oob)
    g.y = (-1.0f*py[2].value + 1.0f*py[3].value)/1.0f;

  else if(!py[1].oob && !py[2].oob)
    g.y = (-1.0f*py[1].value + 1.0f*py[2].value)/1.0f;

  return g;
}

template<typename T>
glm::vec3 normal_impl(const soil::buffer& buffer, const soil::index index, glm::ivec2 p){

  auto _buffer = buffer.as<T>();

  sample_t<T> px[5], py[5];
  for(size_t i = 0; i < 5; ++i){

    const glm::ivec2 pos_x = p + glm::ivec2(-2 + i, 0);
    if(!index.oob<2>(pos_x)){
      px[i].oob = false;
      px[i].pos = pos_x;

      const size_t ind = index.flatten<2>(pos_x);
      px[i].value = _buffer[ind];
    }
  
    const glm::ivec2 pos_y = p + glm::ivec2(0, -2 + i);
    if(!index.oob<2>(pos_y)){
      py[i].oob = false;
      py[i].pos = pos_y;

      const size_t ind = index.flatten<2>(pos_y);
      py[i].value = _buffer[ind];
    }
  }

  const glm::vec2 g = gradient_detailed<T>(px, py);

  // Normal Vector from Gradient

  glm::vec3 n = glm::vec3(-g.x, 1.0f, -g.y);
  if(length(n) > 0)
    n = normalize(n);
  return n;
}

}

// Surface Normal from Surface Gradient

//! Simple Transform Layer, which takes a 2D Layer
//! that returns some floating point type, and returns
//! the normal vector, either single sample or fully baked.
//!
struct normal {

  normal(soil::index index, const soil::layer& layer):
    index{index}{
      auto cached = std::get<soil::cached>(layer._layer);
      this->buffer = soil::buffer(cached.as<float>().buffer);
    }

  //! Single Sample Value
  glm::vec3 operator()(const glm::ivec2 pos){    
    switch(buffer.type()){
      case soil::FLOAT32: return normal_impl<float>(buffer, index, pos);
      case soil::FLOAT64: return normal_impl<double>(buffer, index, pos);
      default: throw std::invalid_argument("type is not accepted");
    }
  }

  //! Bake a whole buffer!
  //! Note: we make sure that the indexing structure of the buffer is respected.
  soil::buffer full(){

    return soil::indexselect(index.type(), [self=this]<typename T>() -> soil::buffer {

      if constexpr(std::same_as<typename T::vec_t, soil::ivec2>){

        auto index = self->index.as<T>();
        auto out = buffer_t<vec3>{index.elem()};

        for(const auto& pos: index.iter()){
          glm::vec3 n = self->operator()(pos);
          n = { n.x, -n.z, n.y};
          n = 0.5f*n + 0.5f;
          out[index.flatten(pos)] = {n.x, n.y, n.z};
        }

        return std::move(soil::buffer(std::move(out)));

      } else {
        throw std::invalid_argument("can't extract a full noise buffer from a non-2D index");
      }

    });

  }

private:
  soil::index index;
  soil::buffer buffer;
};

} // end of namespace soil

#endif