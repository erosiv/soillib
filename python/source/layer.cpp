#ifndef SOILLIB_PYTHON_LAYER
#define SOILLIB_PYTHON_LAYER

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <soillib/util/types.hpp>

#include <soillib/layer/layer.hpp>

#include <soillib/layer/constant.hpp>
#include <soillib/layer/computed.hpp>

#include <soillib/layer/noise.hpp>
#include <soillib/layer/normal.hpp>

#include "glm.hpp"

//! General Layer Binding Function
void bind_layer(nb::module_& module){

  //
  //
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
    return soil::typeselect(constant.type(), [&constant, &index]<typename T>() -> nb::object {
      T value = constant.as<T>()(index);
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