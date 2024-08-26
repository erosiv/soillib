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

  // Bake a Node!
  node bake(const size_t size){
    return std::visit([size](auto&& args){
      return soil::select(args.type(), [args, size]<typename T>(){
        auto node_t = args.template as<T>();
        auto buffer_t = soil::buffer_t<T>(size);
       // auto cached_t = soil::cached_t<T>(buffer_t);
        for(size_t index = 0; index < size; ++index)
          buffer_t[index] = node_t(index);
        return soil::node(std::move(soil::cached(buffer_t)));
      });
    }, this->_node);
  }

  node_v _node;
};

} // end of namespace soil

#endif