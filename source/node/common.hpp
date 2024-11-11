#ifndef SOILLIB_NODE_CONSTANT
#define SOILLIB_NODE_CONSTANT

#include <soillib/soillib.hpp>
#include <soillib/core/node.hpp>

namespace soil {

//! Here are some basic node implementations.
//!
//! \todo Consider whether the functions should be made static somehow.
//!   That way they don't have live on the ether, and can be 'fixed'.

struct constant {

  template<typename T>
  static soil::node make_node(const T value){

    soil::node_t<T> node_t([value](const size_t index) -> T {
      return value;
    });
    
    return soil::node(std::move(node_t));
  }

};

struct computed {

  template<typename T>
  using func_t = std::function<T(const size_t)>;

  template<typename T>
  static soil::node make_node(const func_t<T> f){

    soil::node_t<T> node_t([f](const size_t index) -> T {
      return f(index);
    });
    
    return soil::node(std::move(node_t));
  }

};

struct cached {

  template<typename T>
  static soil::node make_node(soil::buffer* buffer_p){

    soil::node_t<T> node_t(
      [buffer_p](const size_t i) -> T {
        return buffer_p->as<T>()[i];
      },
      [buffer_p](const size_t i) -> T& {
        return buffer_p->as<T>()[i];
      }
    );

    soil::node node = soil::node(std::move(node_t));
    node.size = buffer_p->elem();
    return node;

  }

};

}

#endif