#ifndef SOILLIB_UTIL_SURFACE
#define SOILLIB_UTIL_SURFACE

#include <soillib/soillib.hpp>
//#include <soillib/util/buf.hpp>
//#include <soillib/util/index.hpp>

/*
additional methods to implement:
- gradient function
- higher detail normal vector (finite differences)
-
*/

namespace soil {
namespace surface {

// some lerp
// now what would make the most sense
// if there was some interface to perform entire
// operations on lerped sets!

template<typename T>
const static inline float height(T& map, vec2 p){

  if(map.oob(p))
    return 0.0f;

  ivec2 m = p;

  if(map.oob(m + ivec2(1)))
    return map.get(m)->height;

  vec2 f = glm::fract(p);
  float h00 = map.get(m + ivec2(0, 0))->height;
  float h01 = map.get(m + ivec2(0, 1))->height;
  float h10 = map.get(m + ivec2(1, 0))->height;
  float h11 = map.get(m + ivec2(1, 1))->height;
  return (1.0f-f.y)*(h00*(1.0f-f.x) + h01*f.x) + f.y*(h10*(1.0f-f.x) + h11*f.x);

}

template<typename T>
const static inline vec3 normal(T& map, ivec2 p, const vec3 s){

  vec3 n = vec3(0, 0, 0);

  vec2 pxa = p;
  if(!map.oob(p - ivec2(1, 0)))
    pxa -= ivec2(1, 0);

  vec2 pxb = p;
  if(!map.oob(p + ivec2(1, 0)))
    pxb += ivec2(1, 0);

  vec2 pya = p;
  if(!map.oob(p - ivec2(0, 1)))
    pya -= ivec2(0, 1);

  vec2 pyb = p;
  if(!map.oob(p + ivec2(0, 1)))
    pyb += ivec2(0, 1);

  // Compute Gradient

  n.x = -s.y*(map.get(pxb)->height - map.get(pxa)->height)/length(pxb-pxa);
  n.y = 1.0f;
  n.z = -s.y*(map.get(pyb)->height - map.get(pya)->height)/length(pyb-pya);
  n = n;

  if(length(n) > 0)
    n = normalize(n);
  return n;

}

/*
template<typename T>
const static inline vec3 normal(T& map, ivec2 p, const vec3 s){

  vec3 n = vec3(0, 0, 0);

  if(!map.oob(p + ivec2( 1, 1)))
    n += cross( s*vec3( 0.0, map.get(p+ivec2( 0, 1))->height - map.get(p)->height, 1.0), s*vec3( 1.0, map.get(p+ivec2( 1, 0))->height - map.get(p)->height, 0.0));

  if(!map.oob(p + ivec2(-1,-1)))
    n += cross( s*vec3( 0.0, map.get(p-ivec2( 0, 1))->height - map.get(p)->height,-1.0), s*vec3(-1.0, map.get(p-ivec2( 1, 0))->height - map.get(p)->height, 0.0));

  //Two Alternative Planes (+X -> -Y) (-X -> +Y)
  if(!map.oob(p + ivec2( 1,-1)))
    n += cross( s*vec3( 1.0, map.get(p+ivec2( 1, 0))->height - map.get(p)->height, 0.0), s*vec3( 0.0, map.get(p-ivec2( 0, 1))->height - map.get(p)->height,-1.0));

  if(!map.oob(p + ivec2(-1, 1)))
    n += cross( s*vec3(-1.0, map.get(p-ivec2( 1, 0))->height - map.get(p)->height, 0.0), s*vec3( 0.0, map.get(p+ivec2( 0, 1))->height - map.get(p)->height, 1.0));

  if(length(n) > 0)
    n = normalize(n);
  return n;

}
*/

/*

const inline vec3 normal(const ivec2 p){

  vec3 n = vec3(0, 0, 0);
  const vec3 s = vec3(1.0, World::mapscale, 1.0);

  if(!oob(p + ivec2( 1, 1)))
    n += cross( s*vec3( 0.0, height(p+ivec2( 0, 1)) - height(p), 1.0), s*vec3( 1.0, height(p+ivec2( 1, 0)) - height(p), 0.0));

  if(!oob(p + ivec2(-1,-1)))
    n += cross( s*vec3( 0.0, height(p-ivec2( 0, 1)) - height(p),-1.0), s*vec3(-1.0, height(p-ivec2( 1, 0)) - height(p), 0.0));

  //Two Alternative Planes (+X -> -Y) (-X -> +Y)
  if(!oob(p + ivec2( 1,-1)))
    n += cross( s*vec3( 1.0, height(p+ivec2( 1, 0)) - height(p), 0.0), s*vec3( 0.0, height(p-ivec2( 0, 1)) - height(p),-1.0));

  if(!oob(p + ivec2(-1, 1)))
    n += cross( s*vec3(-1.0, height(p-ivec2( 1, 0)) - height(p), 0.0), s*vec3( 0.0, height(p+ivec2( 0, 1)) - height(p), 1.0));

  if(length(n) > 0)
    n = normalize(n);
  return n;

}

*/

};  // end of namespace surface
};  // end of namespace soil

#endif
