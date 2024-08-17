#ifndef SOILLIB_PYTHON_LAYER
#define SOILLIB_PYTHON_LAYER

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <soillib/layer/layer.hpp>
#include <soillib/layer/constant.hpp>
#include <soillib/layer/noise.hpp>
#include <soillib/layer/normal.hpp>

#include "glm.hpp"

//
//
//

//! General Layer Binding Function
void bind_layer(nb::module_& module){

  auto normal = nb::class_<soil::normal>(module, "normal");
  normal.def(nb::init<>());
  normal.def_static("__call__", soil::normal::operator());

  auto constant = nb::class_<soil::constant>(module, "constant");
  constant.def(nb::init<const std::string, const soil::multi&>());
  constant.def("__call__", &soil::constant::operator());

  //
  // Noise Sampler Type
  //

  using noise_t = soil::noise::sampler;
  auto noise = nb::class_<noise_t>(module, "noise");
  noise.def(nb::init<>());
  noise.def("get", &noise_t::get);

}

#endif