#ifndef SOILLIB_PYTHON
#define SOILLIB_PYTHON

// soillib Python Bindings
// Nicholas McDonald 2024

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <memory>

// Bind Function Declarations

void bind_util(py::module& module);

// Module Main Function

PYBIND11_MODULE(soillib, module){

module.doc() = "Soillib Python Module";

bind_util(module);

}

#endif