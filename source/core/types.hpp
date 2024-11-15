#ifndef SOILLIB_TYPES
#define SOILLIB_TYPES

#include <format>
#include <glm/gtc/type_ptr.hpp>
#include <soillib/soillib.hpp>
#include <typeinfo>

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

// Compute Device Enumerator

enum host_t {
  CPU,
  GPU
};

template<host_t T>
struct hostdesc;

template<>
struct hostdesc<CPU> {
  static constexpr std::string name = "CPU";
};

template<>
struct hostdesc<GPU> {
  static constexpr std::string name = "GPU";
};

template<typename F, typename... Args>
auto select(const soil::host_t host, F lambda, Args &&...args) {
  switch (host) {
  case soil::CPU:
    return lambda.template operator()<CPU>(std::forward<Args>(args)...);
  case soil::GPU:
    return lambda.template operator()<GPU>(std::forward<Args>(args)...);
  default:
    throw std::invalid_argument("host not supported");
  }
}

// Vector Type Aliases (Convenience)

constexpr auto defaultp = glm::qualifier::packed_highp;

template<size_t D>
using ivec = glm::vec<D, int, defaultp>;
template<size_t D>
using uvec = glm::vec<D, unsigned int, defaultp>;
template<size_t D>
using fvec = glm::vec<D, float, defaultp>;
template<size_t D>
using dvec = glm::vec<D, double, defaultp>;
template<size_t D>
using vec = fvec<D>; // Default Precision

using ivec1 = ivec<1>;
using ivec2 = ivec<2>;
using ivec3 = ivec<3>;
using ivec4 = ivec<4>;

using uvec1 = uvec<1>;
using uvec2 = uvec<2>;
using uvec3 = uvec<3>;
using uvec4 = uvec<4>;

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
template<typename T>
struct typedesc;

template<>
struct typedesc<int> {
  static constexpr std::string name = "int";
  static constexpr dtype type = INT;
};

template<>
struct typedesc<float> {
  static constexpr std::string name = "float";
  static constexpr dtype type = FLOAT32;
};

template<>
struct typedesc<double> {
  static constexpr std::string name = "double";
  static constexpr dtype type = FLOAT64;
};

template<>
struct typedesc<vec2> {
  static constexpr std::string name = "vec2";
  static constexpr dtype type = VEC2;
};

template<>
struct typedesc<vec3> {
  static constexpr std::string name = "vec3";
  static constexpr dtype type = VEC3;
};

template<>
struct typedesc<ivec2> {
  static constexpr std::string name = "ivec2";
  static constexpr dtype type = IVEC2;
};

template<>
struct typedesc<ivec3> {
  static constexpr std::string name = "ivec3";
  static constexpr dtype type = IVEC3;
};

// Enum-Based Runtime Polymorphic Visitor Pattern:
//
//  Strict-typed, templated implementations of polymorphic
//  classes are defined statically and selected at runtime
//  using an enumerator based type identifier, without inheritance.
//
//  The polymorphic wrapper class maintains a pointer to
//  a stub base-class, which can be cast to the correct type
//  on demand, requiring a single type deduction switch call.
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
  virtual ~typedbase() {};
  constexpr virtual soil::dtype type() noexcept {
    return {};
  }
};

} // namespace

template<typename Type, typename F>
struct type_op_error: std::exception {
  type_op_error(F lambda){
    this->msg = std::format("invalid type <{}>: failed to match constraints", typedesc<Type>::name);
  }
  const char *what() const noexcept override {
    return this->msg.c_str();
  }
private:
  std::string msg;
};

// Templated Lambda Concept Matching Interface:
//
//  Since the select method switches the runtime dynamic type enumerator,
//  each possible lambda operator call must compile. Since it possible to
//  pass a concept into the lambda expression, this won't compile if any
//  of the types below doesn't match the concept.
//
//  Therefore, we ,atch based on a derived concept which is satisfied when the
//  lambdas original concept is matched (a lambda meta concept).

template<typename T, typename F, typename... Args>
concept matches_lambda = requires(F lambda, Args &&...args) {
  { lambda.template operator()<T>(std::forward<Args>(args)...) };
};

//! select accepts a type enumerator and a templated lambda,
//! which it subsequently calls with a strict-typed evaluation.
//!
//! this effectively instantiates every required template of the
//! desired lambda expression, and executes the runtime selection.
template<typename F, typename... Args>
auto select(const soil::dtype type, F lambda, Args &&...args) {

  // Note: Separating out the expressions below doesn't work,
  //  because otherwise the type of select_call would be deduced
  //  differently at compile time depending on the position in
  //  the switch-statement, leading to a conflicting return type.
  //
  //  It relies on the if constexpr else paths not participating
  //  in the return type deduction at all, which would be void.
  //
  // // DOESNT WORK!
  // template<typename T, typename F, typename... Args>
  // constexpr auto select_call(F lambda, Args &&...args){
  //   if constexpr(matches_lambda<T, F, Args...>){
  //     return lambda.template operator()<T>(std::forward<Args>(args)...);
  //   } else {
  //     throw soil::type_op_error<T, F>(lambda);
  //   }
  // }

  switch (type) {
    case soil::INT:
      if constexpr(matches_lambda<int, F, Args...>){
        return lambda.template operator()<int>(std::forward<Args>(args)...);
      } else {
        throw soil::type_op_error<int, F>(lambda);
      }
      break;
    case soil::FLOAT32:
      if constexpr(matches_lambda<float, F, Args...>){
        return lambda.template operator()<float>(std::forward<Args>(args)...);
      } else {
        throw soil::type_op_error<float, F>(lambda);
      }
      break;
    case soil::FLOAT64:
      if constexpr(matches_lambda<double, F, Args...>){
        return lambda.template operator()<double>(std::forward<Args>(args)...);
      } else {
        throw soil::type_op_error<double, F>(lambda);
      }
      break;
    case soil::VEC2:
      if constexpr(matches_lambda<vec2, F, Args...>){
        return lambda.template operator()<vec2>(std::forward<Args>(args)...);
      } else {
        throw soil::type_op_error<vec2, F>(lambda);
      }
      break;
    case soil::VEC3:
      if constexpr(matches_lambda<vec3, F, Args...>){
        return lambda.template operator()<vec3>(std::forward<Args>(args)...);
      } else {
        throw soil::type_op_error<vec3, F>(lambda);
      }
      break;
    case soil::IVEC2:
      if constexpr(matches_lambda<ivec2, F, Args...>){
        return lambda.template operator()<ivec2>(std::forward<Args>(args)...);
      } else {
        throw soil::type_op_error<ivec2, F>(lambda);
      }
      break;
    case soil::IVEC3:
      if constexpr(matches_lambda<ivec3, F, Args...>){
        return lambda.template operator()<ivec3>(std::forward<Args>(args)...);
      } else {
        throw soil::type_op_error<ivec3, F>(lambda);
      }
      break;
    default:
      throw std::invalid_argument("type not supported");
  }

}

template<class I>
constexpr bool is_index_2D() { return I::n_dims == 2; }

template<typename I>
concept index_2D = is_index_2D<I>();

enum dindex {
  FLAT1,
  FLAT2,
  FLAT3,
  FLAT4,
  QUAD
};

// base class
struct indexbase {
  virtual ~indexbase() {};
  constexpr virtual soil::dindex type() noexcept {
    return {};
  }
};

} // namespace soil

#endif