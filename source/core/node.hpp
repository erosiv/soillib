#ifndef SOILLIB_NODE
#define SOILLIB_NODE

#include <soillib/soillib.hpp>
#include <soillib/core/types.hpp>

#include <soillib/node/cached.hpp>
#include <soillib/node/computed.hpp>
#include <soillib/node/constant.hpp>

namespace soil {

//! A layer represents constant, stored or computed
//! quantity distributed over the domain.
//!
//! Layers are modular and can be composed to yield
//! user-defined quantities, which are necessary for
//! executing any dynamic transport model.
//! 
//! Layers can also cache and pre-compute results for
//! efficient deferred computation.

using node_v = std::variant<
  soil::cached,
  soil::constant,
  soil::computed
>;

struct node {

  node(){}

  node(const soil::buffer buffer):
    _node{soil::cached{buffer}}{}

  node(soil::cached&& _node):
    _node{_node}{}

  node(soil::constant&& _node):
    _node{_node}{}

  node(soil::computed&& _node):
    _node{_node}{}

  node(node_v&& _node):
    _node{_node}{}

  template<typename T>
  T operator()(const size_t index) {
    return std::visit([index](auto&& args){
      return args.template operator()<T>(index);
    }, this->_node);
  }

  soil::dtype type() const {
    return std::visit([](auto&& args){
      return args.type();
    }, this->_node);
  }

  node_v _node;
};

} // end of namespace soil

#endif