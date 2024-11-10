#ifndef SOILLIB_PYTHON_LAYER
#define SOILLIB_PYTHON_LAYER

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/ndarray.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/function.h>

#include <soillib/core/types.hpp>

#include <soillib/core/node.hpp>

#include <soillib/node/noise.hpp>
#include <soillib/node/normal.hpp>
#include <soillib/node/flow.hpp>

#include <iostream>

#include "glm.hpp"

//! General Node Binding Function
void bind_node(nb::module_& module){

  //
  // Layer Wrapper Type
  //

  auto node = nb::class_<soil::node>(module, "node");
  node.def(nb::init<const soil::buffer>());
  node.def(nb::init<soil::cached&&>());
  node.def(nb::init<soil::computed&&>());

  node.def_prop_ro("type", &soil::node::type);
  node.def_prop_ro("buffer", [](soil::node& node) -> nb::object {
    if(node.dnode() == soil::CACHED){
      auto buffer = node.as<soil::cached>().buffer;
      return nb::cast(buffer);
    }
    else return nb::none();
  }, nb::rv_policy::reference_internal);

  node.def("__call__", [](soil::node& node, const size_t index){
    return soil::select(node.type(), [&node, index]<typename T>() -> nb::object {
      T value = node.template operator()<T>(index);
      return nb::cast<T>(std::move(value));
    });
  });

  node.def("__setitem__", [](soil::node& node, const nb::slice& slice, const nb::object value){

    auto buffer = node.as<soil::cached>().buffer;
    soil::select(buffer.type(), [&buffer, &slice, &value]<typename S>(){

      auto buffer_t = buffer.as<S>();           // Assignable Strict-Type Buffer
      const auto value_t = nb::cast<S>(value);  // Assignable Value

      // Read Slice:
      Py_ssize_t start, stop, step;
      if(PySlice_GetIndices(slice.ptr(), buffer_t.elem(), &start, &stop, &step) != 0)
        throw std::runtime_error("slice is invalid!");
      
      // Assign Values!
      for(int index = start; index < stop; index += step)
        buffer_t[index] = value_t;
    
    });
  });

  //
  // Special Layer-Based Operations
  //  These will be unified and expanded later!
  //

  node.def("track", [](soil::node& lhs, soil::node& rhs, const nb::object _lrate){

    if(lhs.type() != rhs.type())
      throw std::invalid_argument("nodes are not of the same type");

    soil::select(rhs.type(), [&lhs, &rhs, _lrate]<typename T>(){
      if constexpr (std::is_scalar_v<T>) {
        const T lrate = nb::cast<T>(_lrate);
        auto lhs_t = lhs.as<soil::cached>().buffer.as<T>();
        auto rhs_t = rhs.as<soil::cached>().buffer.as<T>();
        for(size_t i = 0; i < lhs_t.elem(); ++i){
          const T lhs_value = lhs_t[i];
          const T rhs_value = rhs_t[i];
          lhs_t[i] = lhs_value * (T(1.0) - lrate) + rhs_value * lrate;
        }
      } else {
        using V = typename T::value_type;
        const V lrate = nb::cast<V>(_lrate);
        auto lhs_t = lhs.as<soil::cached>().buffer.as<T>();
        auto rhs_t = rhs.as<soil::cached>().buffer.as<T>();
        for(size_t i = 0; i < lhs_t.elem(); ++i){
          const T lhs_value = lhs_t[i];
          const T rhs_value = rhs_t[i];
          lhs_t[i] = lhs_value * (V(1.0) - lrate) + rhs_value * lrate;
        }
      }
    });

  });

  //
  //
  //

  auto normal = nb::class_<soil::normal>(module, "normal");
  normal.def(nb::init<const soil::index&, const soil::node&>());
  normal.def("full", [](const soil::normal& normal){
    return soil::node(std::move(normal.full()));
  });

  auto flow = nb::class_<soil::flow>(module, "flow");
  flow.def(nb::init<const soil::index&, const soil::node&>());
  flow.def("__call__", [](const soil::flow& flow){
    return soil::node(std::move(flow.full()));
  });

  auto direction = nb::class_<soil::direction>(module, "direction");
  direction.def(nb::init<const soil::index&, const soil::node&>());
  direction.def("__call__", [](const soil::direction& direction){
    return soil::node(std::move(direction.full()));
  });

  auto accumulation = nb::class_<soil::accumulation>(module, "accumulation");
  accumulation.def(nb::init<const soil::index&, const soil::node&>());
  accumulation.def("__call__", [](const soil::accumulation& accumulation){
    return soil::node(std::move(accumulation.full()));
  });

  accumulation.def_prop_rw("steps", 
  [](const soil::accumulation& accumulation){
    return accumulation.steps;
  },
  [](soil::accumulation& accumulation, const size_t steps){
    accumulation.steps = steps;
  });

  accumulation.def_prop_rw("iterations", 
  [](const soil::accumulation& accumulation){
    return accumulation.iterations;
  },
  [](soil::accumulation& accumulation, const size_t iterations){
    accumulation.iterations = iterations;
  });

  accumulation.def_prop_rw("samples", 
  [](const soil::accumulation& accumulation){
    return accumulation.samples;
  },
  [](soil::accumulation& accumulation, const size_t samples){
    accumulation.samples = samples;
  });

  //
  // Noise Sampler Type
  //

  module.def("noise", soil::make_noise);

  //
  // Buffer Baking...
  // I suppose this could also be done when a buffer is requested,
  // but that hides the explicitness of it not always being available.
  //

  module.def("bake", [](soil::node& node, soil::index& index){
  
    return soil::select(node.dnode(), [&node, &index]<typename S>(){
      if constexpr(std::same_as<S, soil::cached>){
        return node;
      } else {
        const auto node_t = node.as<S>();
        return soil::select(node_t.type(), [&node_t, &index]<typename T>() {  
          auto buffer_t = soil::buffer_t<T>(index.elem());
          for (size_t i = 0; i < buffer_t.elem(); ++i)
            buffer_t[i] = node_t.template as<T>()(i);
          return soil::node(std::move(soil::cached(buffer_t)));
        });
      }
    });
  });

//
// New Node Interface
//

auto _node = nb::class_<soil::_node>(module, "_node", nb::dynamic_attr());
_node.def_prop_ro("type", &soil::_node::type);

_node.def("__call__", [](soil::_node& node, const size_t index){
  return soil::select(node.type(), [&node, index]<typename T>() -> nb::object {
    T value = node.template operator()<T>(index);
    return nb::cast<T>(std::move(value));
  });
});

//
// Constant-Valued Layer
//

module.def("constant", [](const soil::dtype type, const nb::object object){
  return soil::select(type, [type, &object]<typename T>() -> soil::_node {
    
    const T value = nb::cast<T>(object);
    
    using func_t = soil::map_t<T>::func_t;
    using param_t = soil::map_t<T>::param_t;
    const func_t func = [value](const param_t& in, const size_t index) -> T {
      return value;
    };

    soil::map map = soil::map(func);
    return soil::_node(map, {});

  });
});

//
// Generic Computed Layer
//

module.def("computed", [](const soil::dtype type, const nb::callable object){
  return soil::select(type, [type, &object]<typename T>(){

    using func_s_t = std::function<T(const size_t)>;
    using func_t = soil::map_t<T>::func_t;
    using param_t = soil::map_t<T>::param_t;
    const func_s_t f = nb::cast<func_s_t>(object);
    const func_t func = [f](const param_t& in, const size_t index) -> T {
      return f(index);
    };

    soil::map map = soil::map(func);
    return soil::_node(map, {});
  
  });
});

//
// Cache-Valued Layer, i.e. Lookup Table
// Note: Simplify this be extracting the construction to a function.
//

module.def("cached", [](const soil::buffer& buffer) -> nb::object {

  return soil::select(buffer.type(), [&buffer]<typename T>() -> nb::object {

    using func_t = soil::map_t<T>::func_t;
    using param_t = soil::map_t<T>::param_t;

    const soil::buffer_t<T> buffer_t = buffer.as<T>();
    const func_t func = [buffer_t](const param_t& in, const size_t index) -> T {
      return buffer_t[index];
    };

    soil::map map = soil::map(func);
    soil::_node node = soil::_node(map, {});
    auto object = nb::cast(std::move(node));
    nb::setattr(object, "buffer", nb::find(buffer));
    return object;

  });

});

module.def("cached", [](const soil::dtype type, const size_t size){

  return soil::select(type, [&type, size]<typename T>() -> nb::object {

    using func_t = soil::map_t<T>::func_t;
    using param_t = soil::map_t<T>::param_t;

    const soil::buffer buffer(type, size);

    const soil::buffer_t<T> buffer_t = buffer.as<T>();
    const func_t func = [buffer_t](const param_t& in, const size_t index) -> T {
      return buffer_t[index];
    };

    soil::map map = soil::map(func);
    soil::_node node = soil::_node(map, {});
    auto object = nb::cast(std::move(node));

    auto buffer_obj = nb::cast(std::move(buffer));
    nb::setattr(object, "buffer", buffer_obj);
    return object;

  });

});

}

#endif