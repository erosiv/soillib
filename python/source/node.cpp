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


#include <soillib/node/constant.hpp>

#include <iostream>

#include "glm.hpp"

//! General Node Binding Function
void bind_node(nb::module_& module){

//
// New Node Interface
//

auto node = nb::class_<soil::node>(module, "node", nb::dynamic_attr());
node.def_prop_ro("type", &soil::node::type);

node.def("__call__", [](soil::node& node, const size_t index){
  return soil::select(node.type(), [&node, index]<typename T>() -> nb::object {
    T value = node.val<T>(index);
    return nb::cast<T>(std::move(value));
  });
});

/*
node.def_prop_ro("buffer", [](soil::node& node) -> nb::object {
  if(node.dnode() == soil::CACHED){
    auto buffer = node.as<soil::cached>().buffer;
    return nb::cast(buffer);
  }
  else return nb::none();
}, nb::rv_policy::reference_internal);
*/

//
// Special Layer-Based Operations
//  These will be unified and expanded later!
//

node.def("__setitem__", [](soil::node& node, const nb::slice& slice, const nb::object value){

  soil::select(node.type(), [&node, &slice, &value]<typename S>(){

    Py_ssize_t start, stop, step;
    if(PySlice_GetIndices(slice.ptr(), node.size, &start, &stop, &step) != 0)
      throw std::runtime_error("slice is invalid!");
    
    const auto value_t = nb::cast<S>(value);  // Assignable Value
    for(int index = start; index < stop; index += step)
      node.ref<S>(index) = value_t; 
  
  });

});

node.def("track", [](soil::node& lhs, soil::node& rhs, const nb::object _lrate){

  if(lhs.type() != rhs.type())
    throw std::invalid_argument("nodes are not of the same type");

  if(lhs.size != rhs.size)
    throw std::invalid_argument("nodes are not of the same size");

  soil::select(rhs.type(), [&lhs, &rhs, _lrate]<typename T>(){
    if constexpr (std::is_scalar_v<T>) {
      const T lrate = nb::cast<T>(_lrate);
      for(size_t i = 0; i < lhs.size; ++i){
        const T lhs_value = lhs.val<T>(i);
        const T rhs_value = rhs.val<T>(i);
        lhs.ref<T>(i) = lhs_value * (T(1.0) - lrate) + rhs_value * lrate;
      }
    } else {
      using V = typename T::value_type;
      const V lrate = nb::cast<V>(_lrate);
      for(size_t i = 0; i < lhs.size; ++i){
        const T lhs_value = lhs.val<T>(i);
        const T rhs_value = rhs.val<T>(i);
        lhs.ref<T>(i) = lhs_value * (V(1.0) - lrate) + rhs_value * lrate;
      }
    }
  });

});

//
// Constant-Valued Layer
//

module.def("constant", [](const soil::dtype type, const nb::object object){
  // note: value is considered state. how can this be reflected here?
  return soil::select(type, [&object]<typename T>() -> soil::node {
    const T value = nb::cast<T>(object);
    return soil::constant::make_node<T>(value);
  });
});

//
// Noise Sampler Type
//

module.def("noise", [](const soil::index index, const float seed){
  // note: seed is considered state. how can this be reflected here?
  return soil::noise::make_node(index, seed);
});

//
// Normal Map ?
//

module.def("normal", [](const soil::index& index, const soil::node& node){
  return soil::normal::make_node(index, node);
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
    return soil::node(map, {});
  
  });
});

//
// Cache-Valued Layer, i.e. Lookup Table
// Note: Simplify this be extracting the construction to a function.
//

module.def("cached", [](soil::buffer& buffer) -> nb::object {

  return soil::select(buffer.type(), [&buffer]<typename T>() -> nb::object {

    // explicit buffer copy:
    // the copy constructor will in fact just create
    // a reference w. counting.
    // later, in order to delete the buffer, this
    // pointer has to be deleted at the appropriate time.

    soil::buffer* buffer_p = new soil::buffer(buffer);

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
    node.size = buffer.elem();
    auto object = nb::cast(std::move(node));

    auto buffer_obj = nb::cast(*buffer_p);
    nb::setattr(object, "buffer", buffer_obj);
    return object;

  });

});

module.def("cached", [](const soil::dtype type, const size_t size){

  return soil::select(type, [&type, size]<typename T>() -> nb::object {

    using func_t = soil::map_t<T>::func_t;
    using rfunc_t = soil::map_t<T>::rfunc_t;
    using param_t = soil::map_t<T>::param_t;

    soil::buffer* buffer = new soil::buffer(type, size);

    const func_t func = [buffer](const param_t& in, const size_t index) -> T {
      return buffer->as<T>()[index];
    };

    const rfunc_t rfunc = [buffer](const param_t& in, const size_t index) -> T& {
      return buffer->as<T>()[index];
    };

    soil::map map = soil::map(func, rfunc);
    soil::node node = soil::node(map, {});
    node.size = size;
    auto object = nb::cast(std::move(node));

    auto buffer_obj = nb::cast(*buffer);
    nb::setattr(object, "buffer", buffer_obj);
    return object;

  });

});

// note: consider how to implement this deferred using the nodes
// direct computation? immediate evaluation...

module.def("flow", [](const soil::index& index, const soil::buffer& buffer){
  auto flow = soil::flow(index, buffer);
  return flow.full();
});

module.def("direction", [](const soil::index& index, const soil::buffer& buffer){
  auto direction = soil::direction(index, buffer);
  return direction.full();
});

// this should be replaced with something else...
// the noise layer is also "stateful" - how do we handle
// stateful nodes / conversion operations 

auto accumulation = nb::class_<soil::accumulation>(module, "accumulation");
accumulation.def(nb::init<const soil::index&, const soil::buffer&>());
accumulation.def("__call__", [](const soil::accumulation& accumulation){
  return accumulation.full();
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
// Buffer Baking...
// I suppose this could also be done when a buffer is requested,
// but that hides the explicitness of it not always being available.
//

module.def("bake", [](soil::node& node, soil::index& index){
  return soil::select(node.type(), [&node, &index]<typename T>() {  
    auto buffer_t = soil::buffer_t<T>(index.elem());
    for (size_t i = 0; i < buffer_t.elem(); ++i)
      buffer_t[i] = node.val<T>(i);
    return soil::buffer(std::move(buffer_t));
  });
});

}

#endif