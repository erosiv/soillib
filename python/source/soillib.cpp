#ifndef SOILLIB_PYTHON
#define SOILLIB_PYTHON

// soillib Python Bindings
// Nicholas McDonald 2024

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

// Bind Function Declarations

void bind_io(py::module& module);
void bind_util(py::module& module);
void bind_layer(py::module& module);
void bind_matrix(py::module& module);
void bind_particle(py::module& module);

// Module Main Function

PYBIND11_MODULE(soillib, module){

module.doc() = "Soillib Python Module";

bind_io (module);
bind_util(module);
bind_layer(module);
bind_matrix(module);
bind_particle(module);

}

#endif