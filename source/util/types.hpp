#ifndef SOILLIB_UTIL_TYPE
#define SOILLIB_UTIL_TYPE

#include <soillib/soillib.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <format>

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

// Enum-Based Runtime Polymorphic Visitor Pattern:
//
//  Strict-typed, templated implementations of polymorphic
//  classes are defined statically and selected at runtime
//  using an enumerator based type identifier.
//
//  Note that this pattern allows for receiving the
//  strict-typed template parameter through the lambda
//  for static constexpr checks and control flow based
//  on if constexpr expressions and concepts.
//
//  Additionally, the number of required virtual function
//  implementations is reduced to one, and the number of calls
//  per actual call never exceeds one. This is effectively
//  a variation on the visitor pattern, with better control
//  and no need to declare the variant template with all types.

namespace {

#pragma GCC diagnostic ignored "-Wsubobject-linkage"

struct typedbase {
  virtual ~typedbase(){};
  constexpr virtual soil::dtype type() noexcept {
    return {};
  }
};

}

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

enum dindex {
  FLAT1,
  FLAT2,
  FLAT3,
  FLAT4,
  QUAD
};

// base class
struct indexbase{
  virtual ~indexbase(){};
  constexpr virtual soil::dindex type() noexcept {
    return {};
  }
};

}

#endif