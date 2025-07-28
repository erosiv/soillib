#ifndef SOILLIB_PYTHON
#define SOILLIB_PYTHON

// soillib Python Bindings
// Nicholas McDonald 2025

#include <nanobind/nanobind.h>

// Bind Function Declarations

namespace nb = nanobind;
void bind_shape(nb::module_& module);
void bind_tensor(nb::module_& module);
void bind_io(nb::module_& module);
void bind_op(nb::module_& module);
void bind_model(nb::module_& module);
void bind_util(nb::module_& module);

// Module Main Function

NB_MODULE(MODULE_NAME, module){

nb::set_leak_warnings(false);

module.doc() = "Soillib Python Module";

bind_shape(module);
bind_tensor(module);
bind_io(module);
bind_op(module);
bind_model(module);
bind_util(module);

}

#endif