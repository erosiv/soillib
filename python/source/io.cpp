#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <soillib/io/tiff.hpp>
#include <soillib/io/geotiff.hpp>

void bind_io(py::module& module){

//! TIFF Datatype

auto tiff = py::class_<soil::io::tiff>(module, "tiff");
tiff.def(py::init<const char*>());

tiff.def_readonly("width", &soil::io::tiff::width);
tiff.def_readonly("height", &soil::io::tiff::height);

tiff.def("buf", [](soil::io::tiff& tiff){
  return tiff._buf;
}, py::return_value_policy::reference);

//! GeoTIFF Datatype

auto geotiff = py::class_<soil::io::geotiff, soil::io::tiff>(module, "geotiff");
geotiff.def(py::init<const char*>());

geotiff.def_readonly("width", &soil::io::geotiff::width);
geotiff.def_readonly("height", &soil::io::geotiff::height);

geotiff.def("buf", [](soil::io::geotiff& tiff){
  return tiff._buf;
}, py::return_value_policy::reference);

}

#endif