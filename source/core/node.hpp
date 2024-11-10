#ifndef SOILLIB_NODE
#define SOILLIB_NODE

#include <soillib/core/index.hpp>
#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/soillib.hpp>

#include <soillib/util/error.hpp>
#include <functional>

namespace soil {

// Note: Nodes in general need to be overhauled, so that they
// act like "nodes" in the expected sense of being composable.
// They can still be dynamically typed, but their composition
// and mapping concept needs to be adjusted for flexibility.
// 
// In particular, a decision has to be made about how "compute"
// nodes are intended to function when dependent on multiple nodes.
//
// The concept should be elegant enough that it fits in this
// single header, so that the rest of the "node" code is simply
// implementations of various maps. There should not be individual
// types exposed to python for the various node implementations -
// there should just be functions which return node types or maps
// directly.

struct cached: nodebase {

  cached() {}
  cached(const soil::buffer buffer): buffer{buffer} {}

  constexpr soil::dnode node() noexcept override {
    return soil::CACHED;
  }

  //! retrieve the strict-typed type enumerator
  inline soil::dtype type() const noexcept {
    return this->buffer.type();
  }

  //! check if the contained type is of type
  inline bool is(const soil::dtype type) const noexcept {
    return this->type() == type;
  }

  //! templated lookup operator (cast to T)
  //!
  //! Note that this performs a cast to the desired type.
  //! A static check is performed to guarantee that the
  //! cast of the actual internal type is valid.
  template<typename T>
  T operator()(const size_t index) const {
    return this->buffer.operator[]<T>(index);
  }

  template<typename T>
  T& operator()(const size_t index){
    return this->buffer.operator[]<T>(index);
  }

  soil::buffer buffer;
};

//! computed_t is a strict-typed layer,
//! which has a functor state which is
//! evaluated when indexed.
//!
//! This type is used to implement complex
//! layer types, including coupled layers.
//!
template<typename T>
struct computed_t: typedbase {

  typedef std::function<T(const size_t)> func_t;

  computed_t(func_t func): func(func) {}

  constexpr soil::dtype type() noexcept override {
    return soil::typedesc<T>::type;
  }

  T operator()(const size_t index) const noexcept {
    return this->func(index);
  }

private:
  func_t func;
};

//! computed is a dynamically typed computed layer.
//!
struct computed: nodebase {

  template<typename T>
  using func_t = std::function<T(const size_t)>;

  computed() {}

  template<typename T>
  computed(const soil::dtype type, func_t<T> func): impl{make<T>(type, func)} {}

  constexpr soil::dnode node() noexcept override {
    return soil::COMPUTED;
  }

  //! retrieve the strict-typed type enumerator
  inline soil::dtype type() const noexcept {
    return this->impl->type();
  }

  //! check if the contained type is of type
  inline bool is(const soil::dtype type) const noexcept {
    return this->type() == type;
  }

  //! unsafe cast to strict-type
  template<typename T>
  inline computed_t<T> &as() noexcept {
    return static_cast<computed_t<T> &>(*(this->impl));
  }

  //! unsafe cast to strict-type
  template<typename T>
  inline const computed_t<T> &as() const noexcept {
    return static_cast<computed_t<T> &>(*(this->impl));
  }

  template<typename T>
  T operator()(const size_t index) {
    return select(this->type(), [self = this, index]<typename S>() -> T {
      if constexpr (std::convertible_to<S, T>) {
        return (T)self->as<S>().operator()(index);
      } else
        throw soil::error::cast_error<S, T>();
    });
  }

private:
  using ptr_t = std::shared_ptr<typedbase>;
  ptr_t impl; //!< Strict-Typed Implementation Base Pointer

  template<typename T>
  static ptr_t make(const soil::dtype type, func_t<T> func) {
    return select(type, [func]<typename S>() -> ptr_t {
      if constexpr (std::same_as<T, S>) {
        return std::make_shared<soil::computed_t<S>>(func);
      } else if constexpr (std::convertible_to<T, S>) {
        return std::make_shared<soil::computed_t<S>>([func](const size_t index) {
          return (S)func(index);
        });
      } else
        throw soil::error::cast_error<S, T>();
    });
  }
};


template<typename F, typename... Args>
auto select(const soil::dnode type, F lambda, Args &&...args) {
  switch (type) {
    case soil::CACHED:
      return lambda.template operator()<soil::cached>(std::forward<Args>(args)...);
    case soil::COMPUTED:
      return lambda.template operator()<soil::computed>(std::forward<Args>(args)...);
  default:
    throw std::invalid_argument("type not supported");
  }
}

//! A layer represents constant, stored or computed
//! quantity distributed over the domain.
//!
//! Layers are modular and can be composed to yield
//! user-defined quantities, which are necessary for
//! executing any dynamic transport model.
//!
//! Layers can also cache and pre-compute results for
//! efficient deferred computation.

struct node {

  node() {}

  node(const soil::buffer buffer):
   impl{std::make_shared<soil::cached>(buffer)}{}

  node(soil::cached &&_node): impl{std::make_shared<soil::cached>(_node)} {}

  node(soil::computed &&_node): impl{std::make_shared<soil::computed>(_node)} {}

  //! unsafe cast to strict-type
  template<typename T>
  inline T& as() noexcept {
    return static_cast<T&>(*(this->impl));
  }

  template<typename T>
  inline const T& as() const noexcept {
    return static_cast<T&>(*(this->impl));
  }

  // Polymorphic Type Deduction

  soil::dnode dnode() const {
    return this->impl->node();
  }

  soil::dtype type() const {
    return soil::select(this->dnode(), [self=this]<typename T>(){
      return self->as<T>().type();
    });
  }

  template<typename T>
  T operator()(const size_t index) {
    return soil::select(this->dnode(), [self=this, index]<typename S>(){
      return self->as<S>().template operator()<T>(index);
    });
  }

private:
  using ptr_t = std::shared_ptr<nodebase>;
  ptr_t impl; //!< Strict-Typed Implementation Base Pointer
};

} // end of namespace soil

#endif