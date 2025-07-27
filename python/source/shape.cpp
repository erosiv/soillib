#ifndef SOILLIB_PYTHON_SHAPE
#define SOILLIB_PYTHON_SHAPE

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <soillib/core/shape.hpp>

//! General Util Binding Function
void bind_shape(nb::module_& module) {

//
// Shape Type Binding
//

auto shape = nb::class_<soil::shape>(module, "shape");

shape.def(nb::init<>());
shape.def(nb::init<int>());
shape.def(nb::init<int, int>());
shape.def(nb::init<int, int, int>());
shape.def(nb::init<int, int, int, int>());

shape.def_ro("dim", &soil::shape::dim);
shape.def_ro("elem", &soil::shape::elem);

shape.def("__getitem__", &soil::shape::operator[]);

}

#endif