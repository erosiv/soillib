#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <soillib/layer/layer.hpp>
#include <soillib/layer/normal.hpp>

//
//
//

//! General Layer Binding Function
void bind_layer(py::module& module){

  auto normal = py::class_<soil::normal>(module, "normal");
  normal.def(py::init<>());
  normal.def_static("__call__", soil::normal::operator());
}

#endif