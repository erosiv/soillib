#ifndef SOILLIB_UTIL_INDEX
#define SOILLIB_UTIL_INDEX

#include <soillib/soillib.hpp>
#include <soillib/external/libmorton/morton.h>

namespace soil {

/************************************************
index is a namespace which contains a number of
possible nD indexing methods, for turning a vec
of indices into a flat index. This includes bound
handling, periodicity etc.
************************************************/

struct index_base{};
template<typename T>
concept index_t = std::is_base_of<index_base, T>::value;

namespace index {

// actual index implementations

struct flat: index_base {
  static inline int flatten(const glm::ivec2 p, const glm::ivec2 s){
    return p.x * s.y + p.y;
  }
  static inline glm::ivec2 unflatten(const int i, const glm::ivec2 s){
    int y = ( i / 1   ) % s.x;
    int x = ( i / s.x ) % s.y;
    return glm::ivec2(x, y);
  }
  const static inline bool oob(const glm::ivec2 p, const::glm::ivec2 s){
    if(p.x >= s.x)  return true;
    if(p.y >= s.y)  return true;
    if(p.x  < 0)      return true;
    if(p.y  < 0)      return true;
    return false;
  }
};

/*
struct flat_toric {

  /*

  // Periodic Boundary Condition

  const inline bool oob(const ivec2 p){
    return false;
  }

  inline T* get(ivec2 p){
    if(root.start == NULL) return NULL;
    while(p.x  < 0)  p.x += res.x;
    while(p.y  < 0)  p.y += res.y;
    while(p.x >= res.x)  p.x -= res.x;
    while(p.y >= res.y)  p.y -= res.y;
    return root.start + math::flatten(p, res);
  }

};
  */


/*

struct morton {

  inline int flatten(glm::ivec2 p, glm::ivec2 s){
    return libmorton::morton2D_32_encode(p.x, p.y);
  }

  static inline glm::ivec2 unflatten(int i, glm::ivec2 s){
      return glm::ivec2(0);//libmorton::morton2D_32_decode(i, s.x, s.y);
  }

};

*/

};
};

#endif

/*

//Functions to do Arraymath with

namespace math {


  // 3D Linear Flatten / Unflatten

  int flatten(ivec3 p, ivec3 s){
    if(!all(lessThan(p, s)) || !all(greaterThanEqual(p, ivec3(0))))
      return -1;
    return p.x*s.y*s.z+p.y*s.z+p.z;
  }

  int flatten(int x, int y, int z, const ivec3& s){
    if( x >= s.x || y >= s.y || z >= s.z || x < 0 || y < 0 || z < 0 )
      return -1;
    return x*s.y*s.z+y*s.z+z;
  }

  ivec3 unflatten(int index, ivec3 s){
    int z = ( index / 1   ) % s.x;
    int y = ( index / s.x ) % s.y;
    int x = ( index / ( s.x * s.y ) );
    return ivec3(x, y, z);
  }

  // Morton Order Indexing

  int cflatten(ivec3 p, ivec3 s){
    return libmorton::morton3D_32_encode(p.x, p.y, p.z);
  }

  ivec3 cunflatten(int i, ivec3 s){
    long unsigned int x, y, z;
    libmorton::morton3D_32_decode(i, x, y, z);
    return ivec3(x, y, z);
  }

  int cflatten(ivec2 p, ivec2 s){
    return libmorton::morton2D_32_encode(p.x, p.y);
  }

  int cflatten(int x, int y, ivec2 s){
    return libmorton::morton2D_32_encode(x, y);
  }

  ivec2 cunflatten(int i, ivec2 s){
    long unsigned int x, y, z;
    libmorton::morton2D_32_decode(i, x, y);
    return ivec2(x, y);
  }

  // Helper Functions

  string tostring(ivec3 v){
    stringstream ss;
    ss << v.x << v.y << v.z;
    return ss.str();
  }

  ivec2 rand2( ivec2 max ){
    return ivec2(rand()%max.x, rand()%max.y);
  }

  ivec3 rand3( ivec3 max ){
    return ivec3(rand()%max.x, rand()%max.y, rand()%max.z);
  }

}

*/
