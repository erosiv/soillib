#ifndef SOILLIB_UTIL_TYPE
#define SOILLIB_UTIL_TYPE

#include <soillib/soillib.hpp>
#include <variant>

namespace soil {

template<typename T, size_t N> 
struct vec_t;

using fvec2 = std::array<float, 2>;
using fvec3 = std::array<float, 3>;

//using fvec3 = vec_t<
//using ivec3 = vec_t

//! typedesc is a generic compile-time type descriptor,
//! which provides common properties like string names,
//! or statically related types for specific purposes. 
//!
template<typename T> struct typedesc;

template<>
struct typedesc<int> {
  typedef int value_t;
  static constexpr const char* name = "int"; 
};

template<>
struct typedesc<float> {
  typedef float value_t;
  static constexpr const char* name = "float"; 
};

template<>
struct typedesc<double> {
  typedef double value_t;
  static constexpr const char* name = "double"; 
};

template<>
struct typedesc<fvec2> {
  typedef fvec2 value_t;
  static constexpr const char* name = "fvec2"; 
};

template<>
struct typedesc<fvec3> {
  typedef fvec3 value_t;
  static constexpr const char* name = "fvec3"; 
};

// variant forwarding:
// multi_t is a template-template that accepts a templated type,
// and returns a variant which is specialized by the base types.
//
// additionally, multi is just the regular base type variant.

using multi = std::variant<
  int, float, double
>;

template<template<class> class V>
using multi_t = std::variant<
  V<int>,
  V<float>,
  V<double>
>;

// ...

}

#endif