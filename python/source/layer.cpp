#ifndef SOILLIB_PYTHON_LAYER
#define SOILLIB_PYTHON_LAYER

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/function.h>

#include <soillib/util/types.hpp>

#include <soillib/layer/layer.hpp>

#include <soillib/layer/cached.hpp>
#include <soillib/layer/constant.hpp>
#include <soillib/layer/computed.hpp>

#include <soillib/layer/algorithm/noise.hpp>
#include <soillib/layer/algorithm/normal.hpp>

#include "glm.hpp"

//! General Layer Binding Function
void bind_layer(nb::module_& module){

  //
  // Layer Wrapper Type
  //

  auto layer = nb::class_<soil::layer>(module, "layer");
  layer.def("type", &soil::layer::type);

  layer.def(nb::init<const soil::buffer>());
  layer.def(nb::init<soil::cached&&>());
  layer.def(nb::init<soil::constant&&>());
  layer.def(nb::init<soil::computed&&>());

  layer.def("buffer", [](soil::layer& layer){
    auto cached = std::get<soil::cached>(layer._layer);
    return soil::typeselect(cached.type(), [&cached]<typename T>() -> soil::buffer {
      return soil::buffer(cached.as<T>().buffer);
    });
  });

  //
  // Cache-Valued Layer, i.e. Lookup Table
  //

  auto cached = nb::class_<soil::cached>(module, "cached");
  cached.def("type", &soil::cached::type);

  cached.def("__init__", [](soil::cached* cached, const soil::buffer buffer){
    new (cached) soil::cached(buffer);
  });
  
  cached.def("__call__", [](soil::cached cached, const size_t index){
    return soil::typeselect(cached.type(), [&cached, index]<typename T>() -> nb::object {
      T value = cached.as<T>()(index);
      return nb::cast<T>(std::move(value));
    });
  });

  cached.def("buffer", [](soil::cached cached){
    return soil::typeselect(cached.type(), [&cached]<typename T>() -> soil::buffer {
      return soil::buffer(cached.as<T>().buffer);
    });
  });

  //
  // Constant-Valued Layer
  //

  auto constant = nb::class_<soil::constant>(module, "constant");
  constant.def("type", &soil::constant::type);

  constant.def("__init__", [](soil::constant* constant, const soil::dtype type, const nb::object object){
    soil::typeselect(type, [type, constant, &object]<typename T>(){
      T value = nb::cast<T>(object);
      new (constant) soil::constant(type, value);
    });
  });
  
  constant.def("__call__", [](soil::constant constant, const size_t index){
    return soil::typeselect(constant.type(), [&constant, index]<typename T>() -> nb::object {
      T value = constant.as<T>()(index);
      return nb::cast<T>(std::move(value));
    });
  });

  //
  // Generic Computed Layer
  //

  auto computed = nb::class_<soil::computed>(module, "computed");
  computed.def("type", &soil::computed::type);

  computed.def("__init__", [](soil::computed* computed, const soil::dtype type, const nb::callable object){
    soil::typeselect(type, [type, computed, object]<typename T>(){
      using func_t = std::function<T(const size_t)>;
      func_t func = nb::cast<func_t>(object);
      new (computed) soil::computed(type, func);
    });
  });

  computed.def("__call__", [](soil::computed computed, const size_t index){
    return soil::typeselect(computed.type(), [&computed, index]<typename T>() -> nb::object {
      T value = computed.as<T>()(index);
      return nb::cast<T>(std::move(value));
    });
  });

  //
  //
  //

  auto normal = nb::class_<soil::normal>(module, "normal");
  normal.def(nb::init<const soil::index&, const soil::layer&>());
  normal.def("full", &soil::normal::full);

  //
  // Noise Sampler Type
  //

  auto noise = nb::class_<soil::noise>(module, "noise");
  noise.def(nb::init<const soil::index, const float>());
  noise.def("full", &soil::noise::full);

  //
  // Special Layer-Based Operations
  //  These will be unified and expanded later!
  //

  layer.def("track_float", [](soil::layer& lhs, soil::layer& rhs, const float lrate){

    auto lhs_t = std::get<soil::cached>(lhs._layer).as<float>().buffer;
    auto rhs_t = std::get<soil::cached>(rhs._layer).as<float>().buffer;

    for(size_t i = 0; i < lhs_t.elem(); ++i){
      float lhs_value = lhs_t[i];
      float rhs_value = rhs_t[i];
      lhs_t[i] = lhs_value * (1.0 - lrate) + rhs_value * lrate;
    }

  });

  layer.def("track_vec2", [](soil::layer& lhs, soil::layer& rhs, const float lrate){

    auto lhs_t = std::get<soil::cached>(lhs._layer).as<soil::vec2>().buffer;
    auto rhs_t = std::get<soil::cached>(rhs._layer).as<soil::vec2>().buffer;

    for(size_t i = 0; i < lhs_t.elem(); ++i){
      soil::vec2 lhs_value = lhs_t[i];
      soil::vec2 rhs_value = rhs_t[i];
      lhs_t[i] = lhs_value * (1.0f - lrate) + rhs_value * lrate;
    }

  });

}

#endif