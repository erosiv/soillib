#ifndef SOILLIB_PYTHON_IO
#define SOILLIB_PYTHON_IO

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "glm.hpp"
namespace py = pybind11;

#include <soillib/io/tiff.hpp>
#include <soillib/io/geotiff.hpp>

void bind_io(py::module& module){

//! TIFF Datatype

auto tiff = py::class_<soil::io::tiff>(module, "tiff");
tiff.def(py::init<const char*>());
tiff.def(py::init<soil::array>());

tiff.def("meta", &soil::io::tiff::meta);
tiff.def("read", &soil::io::tiff::read);
tiff.def("write", &soil::io::tiff::write);

tiff.def_property_readonly("width", &soil::io::tiff::width);
tiff.def_property_readonly("height", &soil::io::tiff::height);

tiff.def("array", &soil::io::tiff::array, py::return_value_policy::reference);

//! GeoTIFF Datatype

auto geotiff = py::class_<soil::io::geotiff, soil::io::tiff>(module, "geotiff");

geotiff.def(py::init<>());
geotiff.def(py::init<const char*>());

geotiff.def("meta", &soil::io::geotiff::meta);
geotiff.def("read", &soil::io::geotiff::read);

geotiff.def_property_readonly("min", [](soil::io::geotiff& geotiff){
  return geotiff.min();
});

geotiff.def_property_readonly("max", [](soil::io::geotiff& geotiff){
  return geotiff.max();
});

geotiff.def_property_readonly("scale", [](soil::io::geotiff& geotiff){
  return geotiff.scale();
});

}

#endif