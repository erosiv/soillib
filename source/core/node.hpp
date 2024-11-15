#ifndef SOILLIB_NODE
#define SOILLIB_NODE

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <soillib/core/types.hpp>
#include <soillib/soillib.hpp>

#include <functional>
#include <soillib/util/error.hpp>

namespace soil {

//! node represents a deferred, cachable polymorphic computation graph.
//!
//! Nodes can represent any mapping from any set of value types to
//! any other value type, and store references to dependencies so
//! that the computation can be deferred, cached and manipulated.
//!
//! This has the advantage that complex models can be constructed
//! at run-time by node composition, and can be passed to the back-end
//! without require re-compilation. The computations themselves are
//! still strict-typed, while shape and type constraints are checked
//! at construction.
//!
//! A node can have data as its input dependency, allowing for the
//! definition of dynamically computed quantities which are dependent
//! on other sources of data.
//!
//! Ultimately, we receive an efficient computation graph that can be
//! flexibly parameterized, manipulated and extended.
//!
//! In the future, this can be used together with autograd concepts.
//!
//! Nodes are constructed using mapping functions.
//! - Applied to a node, the computation is deferred and a node is returned.
//! - Applied to a buffer, the computation is immediate and a buffer is returned.
//!
struct node;

//! node_t is a strict-typed node for transforming data.
//!
template<typename T>
struct node_t: typedbase {

  //!\todo add information about constness / no ref
  //!\todo check if we can make this better w. raw function pointers

  using f_val = std::function<T(const size_t)>;
  using f_ref = std::function<T &(const size_t)>;

  //  template<typename F, typename... Args>
  //  map_t(F lambda, Args &&...args): func(F(std::forward<Args>(args)...)){}

  /*
  template<typename F, typename... Args>
  map(F lambda, Args &&...args){
    auto bind = std::bind(lambda, std::forward<Args>(args)...);
    this->impl = make<std::decay<lambda>::type>(bind);
  //lambda, args...))}{}
  }
  */

  node_t(f_val _val): _val(_val) {}

  node_t(f_val _val, f_ref _ref): _val(_val), _ref(_ref) {}

  constexpr soil::dtype type() noexcept override {
    return soil::typedesc<T>::type;
  }

  T val(const size_t index) const noexcept {
    return this->_val(index);
  }

  T &ref(const size_t index) noexcept {
    return this->_ref(index);
  }

private:
  f_val _val;
  f_ref _ref = [](const size_t index) -> T & {
    throw std::runtime_error("NO BACKPROPAGATION POSSIBLE");
  };
};

struct node {

  template<typename T>
  using func_t = std::function<T(const size_t)>;

  template<typename T>
  using rfunc_t = std::function<T &(const size_t)>;

  node() {}

  template<typename T>
  node(soil::node_t<T> &&node_t) {
    this->impl = std::make_shared<soil::node_t<T>>(node_t);
  }

  template<typename T>
  node(func_t<T> func) {
    this->impl = std::make_shared<soil::node_t<T>>(func);
  }

  template<typename T>
  node(func_t<T> func, rfunc_t<T> rfunc) {
    this->impl = std::make_shared<soil::node_t<T>>(func, rfunc);
  }

  inline soil::dtype type() const noexcept {
    return this->impl->type();
  }

  // Strict Type Casting

  template<typename T>
  inline const node_t<T> &as() const noexcept {
    return static_cast<node_t<T> &>(*(this->impl));
  }

  template<typename T>
  inline node_t<T> &as() noexcept {
    return static_cast<node_t<T> &>(*(this->impl));
  }

  // Value and Reference Retrieval

  //! Sample Value
  template<typename T>
  T val(const size_t index) const {
    return this->as<T>().val(index);
  }

  //! Retrieve Value Reference
  template<typename T>
  T &ref(const size_t index) {
    return this->as<T>().ref(index);
  }

  size_t size = 0;

private:
  using ptr_t = std::shared_ptr<typedbase>;
  ptr_t impl; //!< Strict-Typed Implementation Base Pointer
};

} // end of namespace soil

#endif