#ifndef SOILLIB_NODE
#define SOILLIB_NODE

#include <soillib/core/index.hpp>
#include <soillib/core/types.hpp>
#include <soillib/soillib.hpp>

#include <soillib/node/cached.hpp>
#include <soillib/node/computed.hpp>
#include <soillib/node/constant.hpp>

namespace soil {

template<typename F, typename... Args>
auto select(const soil::dnode type, F lambda, Args &&...args) {
  switch (type) {
    case soil::CACHED:
      return lambda.template operator()<soil::cached>(std::forward<Args>(args)...);
    case soil::CONSTANT:
      return lambda.template operator()<soil::constant>(std::forward<Args>(args)...);
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

  node(soil::constant &&_node): impl{std::make_shared<soil::constant>(_node)} {}

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

  // Call Operator

  template<typename T>
  T operator()(const size_t index) {
    return soil::select(this->dnode(), [self=this, index]<typename S>(){
      return self->as<S>().template operator()<T>(index);
    });
  }

  // Bake a Node!
  node bake(const soil::index index) {

    return soil::select(this->dnode(), [self=this, index]<typename S>(){
      const auto node = self->as<S>();
      return soil::select(node.type(), [node, index]<typename T>() {
        auto node_t = node.template as<T>();
        auto buffer_t = soil::buffer_t<T>(index.elem());
        for (size_t i = 0; i < buffer_t.elem(); ++i)
          buffer_t[i] = node_t(i);
        return soil::node(std::move(soil::cached(buffer_t)));
      });
    });
  }

private:
  using ptr_t = std::shared_ptr<nodebase>;
  ptr_t impl; //!< Strict-Typed Implementation Base Pointer
};

} // end of namespace soil

#endif