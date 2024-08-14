#ifndef SOILLIB_PYTHON_PARTICLE
#define SOILLIB_PYTHON_PARTICLE

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <soillib/particle/particle.hpp>
#include <soillib/particle/water.hpp>

//
// 
//

//! General Layer Binding Function
void bind_particle(py::module& module){

  //auto singular = py::class_<soil::matrix::singular>(module, "singular");
  //singular.def(py::init<>());

}

#endif