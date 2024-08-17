#ifndef SOILLIB_UTIL_TYPE
#define SOILLIB_UTIL_TYPE

#include <soillib/soillib.hpp>
#include <variant>

#include <glm/gtc/type_ptr.hpp>

namespace soil {

// Vector Type Declarations

constexpr auto defaultp = glm::qualifier::packed_highp;

using vec1 = glm::vec<1, float, defaultp>;
using vec2 = glm::vec<2, float, defaultp>;
using vec3 = glm::vec<3, float, defaultp>;
using vec4 = glm::vec<4, float, defaultp>;

using dvec1 = glm::vec<1, double, defaultp>;
using dvec2 = glm::vec<2, double, defaultp>;
using dvec3 = glm::vec<3, double, defaultp>;
using dvec4 = glm::vec<4, double, defaultp>;

using ivec1 = glm::vec<1, int, defaultp>;
using ivec2 = glm::vec<2, int, defaultp>;
using ivec3 = glm::vec<3, int, defaultp>;
using ivec4 = glm::vec<4, int, defaultp>;


// helper type for the visitor #4
template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

//! typedesc is a generic compile-time type descriptor,
//! which provides common properties like string names,
//! or statically related types for specific purposes. 
//!
template<typename T> struct typedesc;

template<> struct typedesc<int> {
  static constexpr const char* name = "int"; 
};

template<> struct typedesc<float> {
  static constexpr const char* name = "float"; 
};

template<> struct typedesc<double> {
  static constexpr const char* name = "double"; 
};

template<> struct typedesc<vec2> {
  static constexpr const char* name = "vec2"; 
};

template<> struct typedesc<vec3> {
  static constexpr const char* name = "vec3"; 
};

// variant forwarding:
// multi_t is a template-template that accepts a templated type,
// and returns a variant which is specialized by the base types.
//
// additionally, multi is just the regular base type variant.

using multi = std::variant<
  int, float, double, vec2, vec3
>;

template<template<class> class V>
using multi_t = std::variant<
  V<int>,
  V<float>,
  V<double>,
  V<vec2>,
  V<vec3>
>;

// ...

}

#endif