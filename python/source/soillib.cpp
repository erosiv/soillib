#ifndef SOILLIB_PYTHON
#define SOILLIB_PYTHON

// soillib Python Bindings
// Nicholas McDonald 2024

#include <nanobind/nanobind.h>

// Bind Function Declarations

namespace nb = nanobind;
void bind_index(nb::module_& module);
void bind_io(nb::module_& module);
void bind_util(nb::module_& module);
void bind_node(nb::module_& module);
void bind_matrix(nb::module_& module);
void bind_model(nb::module_& module);
void bind_particle(nb::module_& module);

// Module Main Function

NB_MODULE(soillib, module){

nb::set_leak_warnings(false);

module.doc() = "Soillib Python Module";

bind_index(module);
bind_io(module);
bind_util(module);
bind_node(module);
bind_matrix(module);
bind_model(module);
bind_particle(module);

}

#endif