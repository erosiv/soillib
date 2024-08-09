#ifndef SOILLIB_UTIL_SURFACE
#define SOILLIB_UTIL_SURFACE

#include <soillib/soillib.hpp>

/*
additional methods to implement:
- gradient function
- higher detail normal vector (finite differences)
-
*/

// some lerp
// now what would make the most sense
// if there was some interface to perform entire
// operations on lerped sets!

namespace soil {
namespace surface {

// Surface Map Constraints

template<typename T>
concept surface_t = requires(T t){
  { t.height(glm::ivec2()) } -> std::same_as<float>;
  { t.oob(glm::ivec2()) } -> std::same_as<bool>;
};

template<surface_t T>
const static inline glm::vec2 gradient(T& map, glm::ivec2 p){

  glm::vec2 pxa = p;
  if(!map.oob(p - glm::ivec2(1, 0)))
    pxa -= glm::ivec2(1, 0);

  glm::vec2 pxb = p;
  if(!map.oob(p + glm::ivec2(1, 0)))
    pxb += glm::ivec2(1, 0);

  glm::vec2 pya = p;
  if(!map.oob(p - glm::ivec2(0, 1)))
    pya -= glm::ivec2(0, 1);

  glm::vec2 pyb = p;
  if(!map.oob(p + glm::ivec2(0, 1)))
    pyb += glm::ivec2(0, 1);

  // Compute Gradient

  glm::vec2 g = glm::vec2(0, 0);
  g.x = (map.height(pxb) - map.height(pxa))/length(pxb-pxa);
  g.y = (map.height(pyb) - map.height(pya))/length(pyb-pya);
  return g;

}

template<surface_t T>
const static inline glm::vec2 gradient_detailed(T& map, glm::ivec2 p){

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

  for(size_t i = 0; i < 5; i++){
    if(!map.oob(px[i].pos)){
      px[i].oob = false;
      px[i].height = map.height(px[i].pos);
    }

    if(!map.oob(py[i].pos)){
      py[i].oob = false;
      py[i].height = map.height(py[i].pos);
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

template<surface_t T>
const static inline glm::vec3 normal(T& map, glm::ivec2 p){

  const glm::vec2 g = gradient_detailed(map, p);
  glm::vec3 n = glm::vec3(-g.x, 1.0f, -g.y);
  if(length(n) > 0)
    n = normalize(n);
  return n;

}

};  // end of namespace surface
};  // end of namespace soil

#endif
