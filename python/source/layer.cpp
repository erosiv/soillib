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

  layer.def(nb::init<soil::cached&&>());
  layer.def(nb::init<soil::constant&&>());
  layer.def(nb::init<soil::computed&&>());

  //
  // Cache-Valued Layer, i.e. Lookup Table
  //

  auto cached = nb::class_<soil::cached>(module, "cached");
  cached.def("type", &soil::cached::type);

  cached.def("__init__", [](soil::cached* cached, const soil::dtype type, soil::array array){
    soil::typeselect(type, [type, cached, &array]<typename T>(){
      soil::array_t<T> array_t = std::get<soil::array_t<T>>(array._array);
      new (cached) soil::cached(type, array_t);
    });
  });
  
  cached.def("__call__", [](soil::cached cached, const size_t index){
    return soil::typeselect(cached.type(), [&cached, index]<typename T>() -> nb::object {
      T value = cached.as<T>()(index);
      return nb::cast<T>(std::move(value));
    });
  });

  cached.def("array", [](soil::cached cached){
    return soil::typeselect(cached.type(), [&cached]<typename T>() -> soil::array {
      return soil::array(cached.as<T>().array);
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
  normal.def(nb::init<>());
  normal.def_static("__call__", soil::normal::operator());

  //
  // Noise Sampler Type
  //

  using noise_t = soil::noise::sampler;
  auto noise = nb::class_<noise_t>(module, "noise");
  noise.def(nb::init<>());
  noise.def("get", &noise_t::get);

}

#endif