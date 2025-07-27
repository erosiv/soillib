#ifndef SOILLIB_PYTHON_SHAPE
#define SOILLIB_PYTHON_SHAPE

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <soillib/core/shape.hpp>
#include <format>

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

shape.def("__repr__", [](const soil::shape& shape){
  switch(shape.dim){
    case 1:
      return std::format("soil.shape({})", shape[0]).c_str(); 
    case 2:
      return std::format("soil.shape({}, {})", shape[0], shape[1]).c_str(); 
    case 3:
      return std::format("soil.shape({}, {}, {})", shape[0], shape[1], shape[2]).c_str(); 
    case 4:
      return std::format("soil.shape({}, {}, {}, {})", shape[0], shape[1], shape[2], shape[3]).c_str(); 
  }
});

}

#endif