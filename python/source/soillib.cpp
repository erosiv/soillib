#ifndef SOILLIB_PYTHON
#define SOILLIB_PYTHON

// soillib Python Bindings
// Nicholas McDonald 2024

#include <nanobind/nanobind.h>

// Bind Function Declarations

namespace nb = nanobind;
void bind_io(nb::module_& module);
void bind_util(nb::module_& module);
void bind_layer(nb::module_& module);
// void bind_matrix(nb::module_& module);
// void bind_particle(nb::module_& module);

// Module Main Function

NB_MODULE(soillib, module){

module.doc() = "Soillib Python Module";

bind_io (module);
bind_util(module);
bind_layer(module);
/*
bind_matrix(module);
bind_particle(module);
*/

}

#endif