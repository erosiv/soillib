#ifndef SOILLIB_PYTHON_LAYER
#define SOILLIB_PYTHON_LAYER

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <soillib/layer/layer.hpp>
#include <soillib/layer/normal.hpp>
#include <soillib/layer/const.hpp>

//
//
//

//! General Layer Binding Function
void bind_layer(py::module& module){

  auto normal = py::class_<soil::normal>(module, "normal");
  normal.def(py::init<>());
  normal.def_static("__call__", soil::normal::operator());

  auto layer_const = py::class_<soil::layer_const>(module, "const");
  layer_const.def(py::init<const std::string, const soil::multi&>());
  layer_const.def("__call__", &soil::layer_const::operator());

}

#endif