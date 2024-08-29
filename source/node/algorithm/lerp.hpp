#ifndef SOILLIB_LAYER_LERP
#define SOILLIB_LAYER_LERP

#include <soillib/soillib.hpp>

namespace soil {

/*
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
*/

} // end of namespace soil

#endif