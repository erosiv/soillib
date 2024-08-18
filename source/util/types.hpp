#ifndef SOILLIB_UTIL_TYPE
#define SOILLIB_UTIL_TYPE

#include <soillib/soillib.hpp>
#include <variant>
#include <format>

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

// Type Descriptor Enumerator

enum dtype {
  INT,
  INT32,
  INT64,
  FLOAT,
  FLOAT32,
  FLOAT64,
  VEC2,
  VEC3,
  VEC4,
  IVEC2,
  IVEC3,
  IVEC4,
  DVEC2,
  DVEC3,
  DVEC4
};

//! typedesc is a generic compile-time type descriptor,
//! which provides common properties like string names,
//! or statically related types for specific purposes. 
//!
template<typename T> struct typedesc;

template<> struct typedesc<int> {
  static constexpr std::string name = "int"; 
  static constexpr dtype type = INT;
};

template<> struct typedesc<float> {
  static constexpr std::string name = "float"; 
  static constexpr dtype type = FLOAT32;
};

template<> struct typedesc<double> {
  static constexpr std::string name = "double"; 
  static constexpr dtype type = FLOAT64;
};

template<> struct typedesc<vec2> {
  static constexpr std::string name = "vec2"; 
  static constexpr dtype type = VEC2;
};

template<> struct typedesc<vec3> {
  static constexpr std::string name = "vec3";
  static constexpr dtype type = VEC3;
};

// Error Messages
//! \todo this is exemplary, clean this up.

namespace error {

template<typename From, typename To>
struct cast_error {
  static std::invalid_argument operator()(){
    return std::invalid_argument(value());
  }
  static std::string value(){
    return std::format("invalid cast from <{}> to <{}>", typedesc<From>::name, typedesc<To>::name);
  }
};

}

// I have a nagging feeling that I should be using concepts again...

/*
With this polymorphism idiom:

- Implementations are Strict-Typed
- Implementations derive from a type which provide the typing interface.
- A wrapper class stores a pointer to the arbitrary typed guy,
  and after checking is static_cast to the correct type.
- Retrieving the actual type requires a single virtual function call,
  subsequently requires a switch-case for the type and then the rest
  is actually strict-typed using the lambda concept.
  It's like an STD visit, but I actually have a template parameter
  instead of just a lambda that requires each type to implement smth.
- If I know the type, I can just get the guy directly.

*/


namespace {

#pragma GCC diagnostic ignored "-Wsubobject-linkage"

struct typedbase {
  constexpr virtual soil::dtype type() const noexcept {
    return {};
  }
};

}

//! Templated Visitor Selector
template<class... Ts>
struct overloaded: Ts... { 
  using Ts::operator()...; 
};

//! typeselect accepts a type enumerator and a templated lambda,
//! which it subsequently calls with a strict-typed evaluation.
//!
//! this effectively instantiates every required template of the
//! desired lambda expression, and executes the runtime selection.
template<typename F, typename... Args>
auto typeselect(const soil::dtype type, F lambda, Args&&... args){
  switch(type){
    case soil::INT:     return lambda.template operator()<int>    (std::forward<Args>(args)...);
    case soil::FLOAT32: return lambda.template operator()<float>  (std::forward<Args>(args)...);
    case soil::FLOAT64: return lambda.template operator()<double> (std::forward<Args>(args)...);
    case soil::VEC2:    return lambda.template operator()<vec2>   (std::forward<Args>(args)...);
    case soil::VEC3:    return lambda.template operator()<vec3>   (std::forward<Args>(args)...);
    default: throw std::invalid_argument("type not supported");
  }
}

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