#ifndef SOILLIB_PYTHON
#define SOILLIB_PYTHON

// soillib Python Bindings
// Nicholas McDonald 2025

#include <nanobind/nanobind.h>

// Bind Function Declarations

namespace nb = nanobind;
void bind_index(nb::module_& module);
void bind_buffer(nb::module_& module);
void bind_io(nb::module_& module);
void bind_op(nb::module_& module);
void bind_util(nb::module_& module);

// Module Main Function

NB_MODULE(soillib, module){

nb::set_leak_warnings(false);

module.doc() = "Soillib Python Module";

bind_buffer(module);
bind_index(module);
bind_io(module);
bind_op(module);
bind_util(module);

}

#endif