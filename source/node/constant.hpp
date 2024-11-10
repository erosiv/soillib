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

    using func_t = soil::map_t<T>::func_t;
    using param_t = soil::map_t<T>::param_t;
    const func_t func = [value](const param_t& in, const size_t index) -> T {
      return value;
    };

    soil::map map = soil::map(func);
    return soil::node(map, {});

  }

};

}

#endif