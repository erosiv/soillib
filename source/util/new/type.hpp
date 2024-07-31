#ifndef SOILLIB_UTIL_TYPE
#define SOILLIB_UTIL_TYPE

#include <soillib/soillib.hpp>

namespace soil {

//! typedesc is a generic compile-time type descriptor,
//! which provides common properties like string names,
//! or statically related types for specific purposes. 
//!
template<typename T> struct typedesc;

template<>
struct typedesc<float> {
  typedef float value_t;
  const char* name = "float"; 
};

// ...

}

#endif