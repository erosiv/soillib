#ifndef SOILLIB_PYTHON_LAYER
#define SOILLIB_PYTHON_LAYER

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <soillib/layer/layer.hpp>
#include <soillib/layer/constant.hpp>
#include <soillib/layer/noise.hpp>
#include <soillib/layer/normal.hpp>

//
//
//

//! General Layer Binding Function
void bind_layer(py::module& module){

  auto normal = py::class_<soil::normal>(module, "normal");
  normal.def(py::init<>());
  normal.def_static("__call__", soil::normal::operator());

  auto constant = py::class_<soil::constant>(module, "constant");
  constant.def(py::init<const std::string, const soil::multi&>());
  constant.def("__call__", &soil::constant::operator());

  //
  // Noise Sampler Type
  //

  using noise_t = soil::noise::sampler;
  auto noise = py::class_<noise_t>(module, "noise");
  noise.def(py::init<>());
  noise.def("get", &noise_t::get);

}

#endif