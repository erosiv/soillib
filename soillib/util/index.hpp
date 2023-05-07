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

// 2D Flat Index

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

// 2D Periodic Index

struct toric: index_base {
  static inline int flatten(glm::ivec2 p, const glm::ivec2 s){
    while(p.x  < 0)  p.x += s.x;
    while(p.y  < 0)  p.y += s.y;
    while(p.x >= s.x)  p.x -= s.x;
    while(p.y >= s.y)  p.y -= s.y;
    return p.x * s.y + p.y;
  }
  static inline glm::ivec2 unflatten(const int i, const glm::ivec2 s){
    int y = ( i / 1   ) % s.x;
    int x = ( i / s.x ) % s.y;
    return glm::ivec2(x, y);
  }
  const static inline bool oob(const glm::ivec2 p, const::glm::ivec2 s){
    return false;
  }
};

// 2D Morton Order Index

struct morton: index_base {

  static inline int flatten(const glm::ivec2 p, const glm::ivec2 s){
    return libmorton::morton2D_32_encode(p.x, p.y);
  }
  static inline glm::ivec2 unflatten(const int i, const glm::ivec2 s){
    long unsigned int x, y;
    libmorton::morton2D_32_decode(i, x, y);
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

};
};

#endif
