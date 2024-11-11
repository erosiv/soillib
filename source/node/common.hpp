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

    soil::map_t<T> map_t([value](const auto& in, const size_t index) -> T {
      return value;
    });
    
    return soil::node(soil::map(std::move(map_t)), {});
  }

};

struct computed {

  template<typename T>
  using func_t = std::function<T(const size_t)>;

  template<typename T>
  static soil::node make_node(const func_t<T> f){

    soil::map_t<T> map_t([f](const auto& in, const size_t index) -> T {
      return f(index);
    });
    
    return soil::node(soil::map(std::move(map_t)), {});
  }

};

struct cached {

  template<typename T>
  static soil::node make_node(soil::buffer* buffer_p){

    using func_t = soil::map_t<T>::func_t;
    using rfunc_t = soil::map_t<T>::rfunc_t;
    using param_t = soil::map_t<T>::param_t;

    const func_t func = [buffer_p](const param_t& in, const size_t i) -> T {
      return buffer_p->as<T>()[i];
    };

    const rfunc_t rfunc = [buffer_p](const param_t& in, const size_t i) -> T& {
      return buffer_p->as<T>()[i];
    };

    // delete buffer

    soil::map map = soil::map(func, rfunc);
    soil::node node = soil::node(map, {});
    node.size = buffer_p->elem();
    return node;

  }

};

}

#endif