#ifndef SOILLIB_TYPES
#define SOILLIB_TYPES

#include <soillib/soillib.hpp>
#include <soillib/core/vector.hpp>

#include <format>
#include <typeinfo>

namespace soil {
 
//
// Hosts
//

//! Compute Device Enumerator
enum host_t {
  CPU,
  GPU
};

//! Compute Device Type Descriptor
template<host_t T>
struct hostdesc;

template<>
struct hostdesc<CPU> {
  static constexpr const char* name = "CPU";
};

template<>
struct hostdesc<GPU> {
  static constexpr const char* name = "GPU";
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

//
// Types
//

//! Type Descriptor Enumerator
enum dtype {
  NONE,
  INT,
  INT32,
  INT64,
  FLOAT,
  FLOAT32,
  FLOAT64
};

//! typedesc is a generic compile-time type descriptor,
//! which provides common properties like string names,
//! or statically related types for specific purposes.
//!
template<typename T>
struct typedesc {
  static constexpr const char* name = "none";
  static constexpr dtype type = NONE;
  typedef void value_t;
};

template<>
struct typedesc<int> {
  static constexpr const char* name = "int";
  static constexpr dtype type = INT;
  typedef int value_t;
};

template<>
struct typedesc<float> {
  static constexpr const char* name = "float";
  static constexpr dtype type = FLOAT32;
  typedef float value_t;
};

template<>
struct typedesc<double> {
  static constexpr const char* name = "double";
  static constexpr dtype type = FLOAT64;
  typedef double value_t;
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

struct typedbase {
  virtual ~typedbase() {};
  constexpr virtual soil::dtype type() noexcept {
    return {};
  }
};

} // namespace

template<typename Type, typename F>
struct type_op_error: std::exception {
  type_op_error(F lambda) {
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
    if constexpr (matches_lambda<int, F, Args...>) {
      return lambda.template operator()<int>(std::forward<Args>(args)...);
    } else {
      throw soil::type_op_error<int, F>(lambda);
    }
    break;
  case soil::FLOAT32:
    if constexpr (matches_lambda<float, F, Args...>) {
      return lambda.template operator()<float>(std::forward<Args>(args)...);
    } else {
      throw soil::type_op_error<float, F>(lambda);
    }
    break;
  case soil::FLOAT64:
    if constexpr (matches_lambda<double, F, Args...>) {
      return lambda.template operator()<double>(std::forward<Args>(args)...);
    } else {
      throw soil::type_op_error<double, F>(lambda);
    }
    break;
  default:
    throw std::invalid_argument("type not supported");
  }
}

} // namespace soil

#endif