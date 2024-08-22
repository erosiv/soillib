#ifndef SOILLIB_TYPES
#define SOILLIB_TYPES

#include <soillib/soillib.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <format>

namespace soil {

// Type Descriptor Enumerator

enum dtype {
  INT,
  INT32,
  INT64,
  FLOAT,
  FLOAT32,
  FLOAT64,
  //! \todo eliminate all vector types
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

// Vector Type Aliases (Convenience)

constexpr auto defaultp = glm::qualifier::packed_highp;

template<size_t D> using ivec = glm::vec<D, int, defaultp>;
template<size_t D> using fvec = glm::vec<D, float, defaultp>;
template<size_t D> using dvec = glm::vec<D, double, defaultp>;
template<size_t D> using vec = fvec<D>; // Default Precision

using ivec1 = ivec<1>;
using ivec2 = ivec<2>;
using ivec3 = ivec<3>;
using ivec4 = ivec<4>;

using fvec1 = fvec<1>;
using fvec2 = fvec<2>;
using fvec3 = fvec<3>;
using fvec4 = fvec<4>;

using dvec1 = dvec<1>;
using dvec2 = dvec<2>;
using dvec3 = dvec<3>;
using dvec4 = dvec<4>;

using vec1 = vec<1>;
using vec2 = vec<2>;
using vec3 = vec<3>;
using vec4 = vec<4>;

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