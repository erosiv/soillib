#ifndef SOILLIB_PYTHON_IO
#define SOILLIB_PYTHON_IO

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/stl/string.h>

#include <soillib/io/tiff.hpp>
#include <soillib/io/geotiff.hpp>

#include "glm.hpp"

void bind_io(nb::module_& module){

//! TIFF Datatype

auto tiff = nb::class_<soil::io::tiff>(module, "tiff");
tiff.def(nb::init<const char*>());
tiff.def(nb::init<soil::buffer, soil::index>());

tiff.def("meta", &soil::io::tiff::meta);
tiff.def("read", &soil::io::tiff::read);
tiff.def("write", &soil::io::tiff::write);

tiff.def_prop_ro("width", &soil::io::tiff::width);
tiff.def_prop_ro("height", &soil::io::tiff::height);

tiff.def("buffer", &soil::io::tiff::buffer, nb::rv_policy::reference);

//! GeoTIFF Datatype

auto geotiff = nb::class_<soil::io::geotiff, soil::io::tiff>(module, "geotiff");

geotiff.def(nb::init<>());
geotiff.def(nb::init<const char*>());

geotiff.def("meta", &soil::io::geotiff::meta);
geotiff.def("read", &soil::io::geotiff::read);

geotiff.def_prop_ro("min", [](soil::io::geotiff& geotiff){
  return geotiff.min();
});

geotiff.def_prop_ro("max", [](soil::io::geotiff& geotiff){
  return geotiff.max();
});

geotiff.def_prop_ro("scale", [](soil::io::geotiff& geotiff){
  return geotiff.scale();
});

}

#endif