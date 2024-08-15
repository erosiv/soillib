#ifndef SOILLIB_PYTHON_MATRIX
#define SOILLIB_PYTHON_MATRIX

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "glm.hpp"
namespace py = pybind11;

#include <soillib/matrix/matrix.hpp>

//
//
//

//! General Layer Binding Function
void bind_matrix(py::module& module){

  auto singular = py::class_<soil::matrix::singular>(module, "singular");
  singular.def(py::init<>());

}

#endif