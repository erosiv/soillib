#ifndef SOILLIB_LAYER_NORMAL
#define SOILLIB_LAYER_NORMAL

#include <soillib/soillib.hpp>
#include <soillib/util/array.hpp>
#include <soillib/layer/layer.hpp>

namespace soil {

namespace {

template<typename T>
glm::vec2 gradient_detailed(const soil::array& array, glm::ivec2 p){

  // Generate the Finite Difference Samples

  struct Point {
    glm::ivec2 pos;
    bool oob = true;
    T height;
  } px[5], py[5];

  px[0].pos = p + glm::ivec2(-2, 0);
  px[1].pos = p + glm::ivec2(-1, 0);
  px[2].pos = p + glm::ivec2( 0, 0);
  px[3].pos = p + glm::ivec2( 1, 0);
  px[4].pos = p + glm::ivec2( 2, 0);

  py[0].pos = p + glm::ivec2( 0,-2);
  py[1].pos = p + glm::ivec2( 0,-1);
  py[2].pos = p + glm::ivec2( 0, 0);
  py[3].pos = p + glm::ivec2( 0, 1);
  py[4].pos = p + glm::ivec2( 0, 2);

  auto _array = std::get<soil::array_t<T>>(array._array);
  auto _shape = _array.shape();

  auto sample = [&](const glm::ivec2 pos) -> T {
    const size_t index = _shape.template flat<2>({(size_t)pos.x, (size_t)pos.y});
    return _array[index];
  };

  for(size_t i = 0; i < 5; i++){

    auto pos_x = px[i].pos;
    auto pos_y = py[i].pos;

    if(!_shape.template oob<2>({(size_t)pos_x.x, (size_t)pos_x.y})){
      px[i].oob = false;
      px[i].height = sample(px[i].pos);
    }
  
    if(!_shape.template oob<2>({(size_t)pos_y.x, (size_t)pos_y.y})){
      py[i].oob = false;
      py[i].height = sample(py[i].pos);
    }
  }

  // Compute Gradient

  glm::vec2 g = glm::vec2(0, 0);

  // X-Element  
  if(!px[0].oob && !px[4].oob)
    g.x = (1.0f*px[0].height - 8.0f*px[1].height + 8.0f*px[3].height - 1.0f*px[4].height)/12.0f;

  else if(!px[0].oob && !px[3].oob)
    g.x = (1.0f*px[0].height - 6.0f*px[1].height + 3.0f*px[2].height + 2.0f*px[3].height)/6.0f;

  else if(!px[0].oob && !px[2].oob)
    g.x = (1.0f*px[0].height - 4.0f*px[1].height + 3.0f*px[2].height)/2.0f;

  else if(!px[1].oob && !px[4].oob)
    g.x = (-2.0f*px[1].height - 3.0f*px[2].height + 6.0f*px[3].height - 1.0f*px[4].height)/6.0f;

  else if(!px[2].oob && !px[4].oob)
    g.x = (-3.0f*px[2].height + 4.0f*px[3].height - 1.0f*px[4].height)/2.0f;

  else if(!px[1].oob && !px[3].oob)
    g.x = (-1.0f*px[1].height + 1.0f*px[3].height)/2.0f;

  else if(!px[2].oob && !px[3].oob)
    g.x = (-1.0f*px[2].height + 1.0f*px[3].height)/1.0f;

  else if(!px[1].oob && !px[2].oob)
    g.x = (-1.0f*px[1].height + 1.0f*px[2].height)/1.0f;

  // Y-Element

  if(!py[0].oob && !py[4].oob)
    g.y = (1.0f*py[0].height - 8.0f*py[1].height + 8.0f*py[3].height - 1.0f*py[4].height)/12.0f;

  else if(!py[0].oob && !py[3].oob)
    g.y = (1.0f*py[0].height - 6.0f*py[1].height + 3.0f*py[2].height + 2.0f*py[3].height)/6.0f;

  else if(!py[0].oob && !py[2].oob)
    g.y = (1.0f*py[0].height - 4.0f*py[1].height + 3.0f*py[2].height)/2.0f;

  else if(!py[1].oob && !py[4].oob)
    g.y = (-2.0f*py[1].height - 3.0f*py[2].height + 6.0f*py[3].height - 1.0f*py[4].height)/6.0f;

  else if(!py[2].oob && !py[4].oob)
    g.y = (-3.0f*py[2].height + 4.0f*py[3].height - 1.0f*py[4].height)/2.0f;

  else if(!py[1].oob && !py[3].oob)
    g.y = (-1.0f*py[1].height + 1.0f*py[3].height)/2.0f;

  else if(!py[2].oob && !py[3].oob)
    g.y = (-1.0f*py[2].height + 1.0f*py[3].height)/1.0f;

  else if(!py[1].oob && !py[2].oob)
    g.y = (-1.0f*py[1].height + 1.0f*py[2].height)/1.0f;

  return g;

}

// Surface Normal from Surface Gradient

template<typename T>
glm::vec3 __normal(const soil::array& array, glm::ivec2 p){

  const glm::vec2 g = gradient_detailed<T>(array, p);
  glm::vec3 n = glm::vec3(-g.x, 1.0f, -g.y);

  if(length(n) > 0)
    n = normalize(n);
  return n;

}

}

/*
Basically, what is happening, is that we have a type that utilizes a function
to execute on an input layer. We could also make it NON-CACHED which would let
us template it further... That would also improve the API.

A cached layer would be something else for later...
But the buffer exists... So we will think about that later.
*/

// In principle, normal can take multiple types.
// This is where concepts become quite useful...

struct normal {

  static soil::array operator()(const soil::array& in){

    soil::shape shape = in.shape();
    array_t<fvec3> out = array_t<fvec3>{shape};

    auto _shape = std::get<soil::shape_t<2>>(shape._shape);
    for(const auto& pos: _shape.iter()){
      const size_t index = _shape.flat(pos);

      glm::vec3 n;
      if(in.type() == "float")
        n = __normal<float>(in, glm::ivec2(pos[0], pos[1]));
      else if(in.type() == "double")
        n = __normal<double>(in, glm::ivec2(pos[0], pos[1]));
      n = { n.x, -n.z, n.y};
      n = 0.5f*n + 0.5f;
      out[index] = {n.x, n.y, n.z};
    }

    return std::move(soil::array(out));
  }

  struct sub {

    static glm::vec3 operator()(const soil::array& in, const glm::ivec2 pos){

      soil::shape shape = in.shape();
      auto _shape = std::get<soil::shape_t<2>>(shape._shape);

      glm::vec3 n;
      if(in.type() == "float")
        n = __normal<float>(in, pos);
      else if(in.type() == "double")
        n = __normal<double>(in, pos);
      //n = { n.x, -n.z, n.y};
      return n;

    }

  };

};

} // end of namespace soil

#endif