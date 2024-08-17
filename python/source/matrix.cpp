#ifndef SOILLIB_PYTHON_MATRIX
#define SOILLIB_PYTHON_MATRIX

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <soillib/matrix/matrix.hpp>

#include "glm.hpp"

//
//
//

//! General Layer Binding Function
void bind_matrix(nb::module_& module){

  auto singular = nb::class_<soil::matrix::singular>(module, "singular");
  singular.def(nb::init<>());

}

#endif