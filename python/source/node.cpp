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
#include <soillib/node/math.hpp>
#include <soillib/node/erosion.hpp>

#include <soillib/node/common.hpp>

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

module.def("computed", [](const soil::dtype type, const nb::callable object){
  return soil::select(type, [&object]<typename T>() -> soil::node {
    const soil::computed::func_t<T> f = nb::cast<soil::computed::func_t<T>>(object);
    return soil::computed::make_node<T>(f);
  });
});

module.def("cached", [](soil::buffer& buffer) -> nb::object {
  soil::buffer* buffer_p = new soil::buffer(buffer);
  auto node = soil::select(buffer.type(), [buffer_p]<typename T>() -> soil::node {
    return soil::cached::make_node<T>(buffer_p);
  });
  auto object = nb::cast(std::move(node));
  auto buffer_obj = nb::cast(*buffer_p);
  nb::setattr(object, "buffer", buffer_obj);
  return object;
});

module.def("cached", [](const soil::dtype type, const size_t size){
  soil::buffer* buffer_p = new soil::buffer(type, size);
  auto node = soil::select(type, [buffer_p]<typename T>() -> soil::node {
    return soil::cached::make_node<T>(buffer_p);
  });
  auto object = nb::cast(std::move(node));
  auto buffer_obj = nb::cast(*buffer_p);
  nb::setattr(object, "buffer", buffer_obj);
  return object;
});

//
// Generic Buffer Reductions
//

module.def("cast", [](const soil::buffer& buf, const soil::dtype type){
  if(buf.type() == type){
    return nb::cast(buf);
  }
  return soil::select(type, [&buf]<std::floating_point To>() -> nb::object {
  return soil::select(buf.type(), [&buf]<std::floating_point From>() -> nb::object {
    soil::buffer buffer = soil::cast<To, From>(buf.as<From>());
    return nb::cast(buffer);
  });
  });
});

module.def("min", [](const soil::buffer& buf){
  return soil::select(buf.type(), [&buf]<std::floating_point S>() -> nb::object {
    return nb::cast(soil::min(buf.as<S>()));
  });
});

module.def("max", [](const soil::buffer& buf){
  return soil::select(buf.type(), [&buf]<std::floating_point S>() -> nb::object {
    return nb::cast(soil::max(buf.as<S>()));
  });
});

//
// Generic Buffer Functions
//

module.def("copy", [](soil::buffer& lhs, const soil::buffer& rhs, soil::vec2 gmin, soil::vec2 gmax, soil::vec2 gscale, soil::vec2 wmin, soil::vec2 wmax, soil::vec2 wscale, float pscale){

  // Note: This supports copy between different buffer types.
  // The interior template selection just requires that the source
  // buffer's type can be converted to the target buffer's type.

  soil::select(lhs.type(), [&]<typename To>(){
    soil::select(rhs.type(), [&]<std::convertible_to<To> From>(){
      soil::copy<To, From>(lhs.as<To>(), rhs.as<From>(), gmin, gmax, gscale, wmin, wmax, wscale, pscale);
    });
  });
});

module.def("set", [](soil::buffer& lhs, const soil::buffer& rhs){
  if(lhs.type() != rhs.type())
    throw soil::error::mismatch_type(lhs.type(), rhs.type());
  soil::select(lhs.type(), [&lhs, &rhs]<typename S>(){
    soil::set<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("set", [](soil::buffer& buffer, const nb::object value){
  soil::select(buffer.type(), [&buffer, &value]<typename S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    soil::set<S>(buffer_t, value_t);
  });
});

module.def("add", [](soil::buffer& lhs, const soil::buffer& rhs){
  if(lhs.type() != rhs.type())
    throw soil::error::mismatch_type(lhs.type(), rhs.type());
  soil::select(lhs.type(), [&lhs, &rhs]<typename S>(){
    soil::add<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("add", [](soil::buffer& buffer, const nb::object value){
  soil::select(buffer.type(), [&buffer, &value]<typename S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    soil::add<S>(buffer_t, value_t);
  });
});

module.def("multiply", [](soil::buffer& lhs, const soil::buffer& rhs){
  if(lhs.type() != rhs.type())
    throw soil::error::mismatch_type(lhs.type(), rhs.type());
  soil::select(lhs.type(), [&lhs, &rhs]<typename S>(){
    soil::multiply<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("multiply", [](soil::buffer& buffer, const nb::object value){
  soil::select(buffer.type(), [&buffer, &value]<typename S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    soil::multiply<S>(buffer_t, value_t);
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

module.def("normal", [](const soil::buffer& buffer, const soil::index& index){
  return soil::normal::operator()(buffer, index);
});

module.def("normal", [](const soil::node& node, const soil::index& index){
  return soil::normal::operator()(node, index);
});


//
// Erosion Kernels
//

auto param_t = nb::class_<soil::param_t>(module, "param_t");
param_t.def(nb::init<>());
param_t.def_rw("maxage", &soil::param_t::maxage);
param_t.def_rw("settling", &soil::param_t::settling);
param_t.def_rw("maxdiff", &soil::param_t::maxdiff);
param_t.def_rw("evapRate", &soil::param_t::evapRate);
param_t.def_rw("depositionRate", &soil::param_t::depositionRate);
param_t.def_rw("entrainment", &soil::param_t::entrainment);
param_t.def_rw("gravity", &soil::param_t::gravity);
param_t.def_rw("momentumTransfer", &soil::param_t::momentumTransfer);
param_t.def_rw("minVol", &soil::param_t::minVol);
param_t.def_rw("lrate", &soil::param_t::lrate);
param_t.def_rw("exitSlope", &soil::param_t::exitSlope);
param_t.def_rw("hscale", &soil::param_t::hscale);

auto model_t = nb::class_<soil::model_t>(module, "model_t");
model_t.def(nb::init<soil::index>());

model_t.def_prop_rw("height",
  [](soil::model_t& model){
    return soil::buffer(model.height);
},[](soil::model_t& model, soil::buffer buffer){
    model.height = buffer.as<float>();
});

model_t.def_prop_rw("suspended",
  [](soil::model_t& model){
    return soil::buffer(model.suspended);
},[](soil::model_t& model, soil::buffer buffer){
    model.suspended = buffer.as<float>();
});

model_t.def_prop_rw("discharge",
  [](soil::model_t& model){
    return soil::buffer(model.discharge);
},[](soil::model_t& model, soil::buffer buffer){
    model.discharge = buffer.as<float>();
});

model_t.def_prop_rw("momentum",
  [](soil::model_t& model){
    return soil::buffer(model.momentum);
},[](soil::model_t& model, soil::buffer buffer){
    model.momentum = buffer.as<soil::vec2>();
});

module.def("gpu_erode", soil::gpu_erode);

// note: consider how to implement this deferred using the nodes
// direct computation? immediate evaluation...

module.def("flow", [](const soil::buffer& buffer, const soil::index& index){
  return soil::flow(buffer, index);
});

module.def("direction", [](const soil::buffer& buffer, const soil::index& index){
  return soil::direction(buffer, index);
});

module.def("accumulation", [](const soil::buffer& buffer, const soil::index& index, int iterations, int samples){
  return soil::accumulation(buffer, index, iterations, samples);
});

module.def("accumulation_weighted", [](const soil::buffer& buffer, const soil::buffer& weights, const soil::index& index, int iterations, int samples){
  return soil::accumulation(buffer, weights, index, iterations, samples);
});

module.def("upstream", [](const soil::buffer& buffer, const soil::index& index, const glm::ivec2 target){
  return soil::upstream(buffer, index, target);
});

module.def("distance", [](const soil::buffer& buffer, const soil::index& index, const glm::ivec2 target){
  return soil::distance(buffer, index, target);
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