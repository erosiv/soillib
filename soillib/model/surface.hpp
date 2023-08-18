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

// Surface Map Function Implementations

template<surface_t T>
const static inline float height(T& map, glm::vec2 p){

  if(map.oob(p))
    return 0.0f;

  glm::ivec2 m = p;

  if(map.oob(m + glm::ivec2(1)))
    return map.height(m);

  glm::vec2 f = glm::fract(p);
  float h00 = map.height(m + glm::ivec2(0, 0));
  float h01 = map.height(m + glm::ivec2(0, 1));
  float h10 = map.height(m + glm::ivec2(1, 0));
  float h11 = map.height(m + glm::ivec2(1, 1));
  return (1.0f-f.y)*(h00*(1.0f-f.x) + h01*f.x) + f.y*(h10*(1.0f-f.x) + h11*f.x);

}


template<surface_t T>
const static inline glm::vec3 subnormal(T& map, glm::ivec2 p){

  glm::vec3 n = glm::vec3(0, 0, 0);

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

  n.x = -(map.subheight(pxb) - map.subheight(pxa))/length(pxb-pxa);
  n.y = 1.0f;
  n.z = -(map.subheight(pyb) - map.subheight(pya))/length(pyb-pya);
  n = n;

  if(length(n) > 0)
    n = normalize(n);
  return n;

}


template<surface_t T>
const static inline glm::vec3 normal(T& map, glm::ivec2 p){

  glm::vec3 n = glm::vec3(0, 0, 0);

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

  n.x = -(map.height(pxb) - map.height(pxa))/length(pxb-pxa);
  n.y = 1.0f;
  n.z = -(map.height(pyb) - map.height(pya))/length(pyb-pya);
  n = n;

  if(length(n) > 0)
    n = normalize(n);
  return n;

}

};  // end of namespace surface
};  // end of namespace soil

#endif
