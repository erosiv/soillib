#ifndef SOILLIB_LAYER_NORMAL
#define SOILLIB_LAYER_NORMAL

#include <soillib/soillib.hpp>
#include <soillib/util/shape.hpp>
#include <soillib/util/array.hpp>
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
glm::vec3 normal_impl(const soil::array& array, const soil::shape shape, glm::ivec2 p){

  auto _array = array.as<T>();

  sample_t<T> px[5], py[5];
  for(size_t i = 0; i < 5; ++i){

    const glm::ivec2 pos_x = p + glm::ivec2(-2 + i, 0);
    if(!shape.oob(pos_x)){
      px[i].oob = false;
      px[i].pos = pos_x;

      const size_t index = shape.flat(pos_x);
      px[i].value = _array[index];
    }
  
    const glm::ivec2 pos_y = p + glm::ivec2(0, -2 + i);
    if(!shape.oob(pos_y)){
      py[i].oob = false;
      py[i].pos = pos_y;

      const size_t index = shape.flat(pos_y);
      py[i].value = _array[index];
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

  normal(const soil::shape& shape, const soil::layer& layer):
    shape{shape}{
      auto cached = std::get<soil::cached>(layer._layer);
      this->array = soil::array(cached.as<float>().array);
    }

  //! Single Sample Value
  glm::vec3 operator()(const glm::ivec2 pos){    
    switch(array.type()){
      case soil::FLOAT32: return normal_impl<float>(array, shape, pos);
      case soil::FLOAT64: return normal_impl<double>(array, shape, pos);
      default: throw std::invalid_argument("type is not accepted");
    }
  }

  //! Bake a whole Array!
  soil::array full(){

    array_t<vec3> out = array_t<vec3>{shape.elem()};
    auto _shape = std::get<soil::shape_t<2>>(shape._shape);
  
    for(const auto& pos: _shape.iter()){
      glm::vec3 n = this->operator()(glm::ivec2(pos[0], pos[1]));
      n = { n.x, -n.z, n.y};
      n = 0.5f*n + 0.5f;
      const size_t index = _shape.flat(pos);
      out[index] = {n.x, n.y, n.z};
    }

    return std::move(soil::array(std::move(out)));
  }

private:
  soil::shape shape;
  soil::array array;
};

} // end of namespace soil

#endif