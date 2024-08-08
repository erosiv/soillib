#ifndef SOILLIB_LAYER_NORMAL
#define SOILLIB_LAYER_NORMAL

#include <soillib/soillib.hpp>
#include <soillib/util/array.hpp>
#include <soillib/layer/layer.hpp>

namespace soil {

namespace {

glm::vec2 gradient_detailed(const soil::array& array, glm::ivec2 p){

  // Generate the Finite Difference Samples

  struct Point {
    glm::ivec2 pos;
    bool oob = true;
    float height;
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

  auto _array = std::get<soil::array_t<float>>(array);
  auto _shape = _array.shape();

  auto sample = [&](const glm::ivec2 pos) -> float {
    const size_t index = _shape.flat<2>({(size_t)pos.x, (size_t)pos.y});
    return _array[index];
  };

  for(size_t i = 0; i < 5; i++){

    auto pos_x = px[i].pos;
    auto pos_y = py[i].pos;

    if(!_shape.oob<2>({(size_t)pos_x.x, (size_t)pos_x.y})){
      px[i].oob = false;
      px[i].height = sample(px[i].pos);
    }
  
    if(!_shape.oob<2>({(size_t)pos_y.x, (size_t)pos_y.y})){
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

//template<surface_t T>
glm::vec3 __normal(const soil::array& array, glm::ivec2 p){

  const glm::vec2 g = gradient_detailed(array, p);
  glm::vec3 n = glm::vec3(-g.x, 1.0f, -g.y);

  if(length(n) > 0)
    n = normalize(n);
  return n;

}

}

struct normal: layer<array_t<float>, array_t<fvec3>> {

  using layer::layer;
  using layer::in_t;
  using layer::out_t;
  using layer::in;

  using layer_t = layer<array_t<float>, array_t<fvec3>>;

  out_t operator()(){

    soil::shape shape = in.shape();
    out_t out = out_t{shape};

    auto _shape = std::get<soil::shape_t<2>>(shape._shape);
    for(const auto& pos: _shape.iter()){
      const size_t index = _shape.flat(pos);
      glm::vec3 n = __normal(in, glm::ivec2(pos[0], pos[1]));
      n = 0.5f*n + 0.5f;
      out[index] = {n.x, n.z, n.y};
    }

    return std::move(out);
  }

};

} // end of namespace soil

#endif