#ifndef SOILLIB_PYTHON_IO
#define SOILLIB_PYTHON_IO

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/stl/string.h>

#include <soillib/io/tiff.hpp>
#include <soillib/io/geotiff.hpp>

#include <soillib/core/node.hpp>

#include "glm.hpp"

void bind_io(nb::module_& module){

//! TIFF Datatype

auto tiff = nb::class_<soil::io::tiff>(module, "tiff");

tiff.def(nb::init<>());
tiff.def(nb::init<const char*>());
tiff.def("__init__", [](soil::io::tiff* tiff, const soil::buffer& buffer, const soil::index& index){
  new (tiff) soil::io::tiff(buffer, index);
});

tiff.def("meta", &soil::io::tiff::meta);
tiff.def("read", &soil::io::tiff::read);
tiff.def("write", &soil::io::tiff::write);

tiff.def_prop_ro("width", &soil::io::tiff::width);
tiff.def_prop_ro("height", &soil::io::tiff::height);

tiff.def_prop_ro("buffer", &soil::io::tiff::buffer, nb::rv_policy::reference_internal);
tiff.def_prop_ro("index", &soil::io::tiff::index);//, nb::rv_policy::reference);

//! GeoTIFF Datatype

auto geotiff = nb::class_<soil::io::geotiff, soil::io::tiff>(module, "geotiff");

geotiff.def(nb::init<>());
geotiff.def(nb::init<const char*>());
geotiff.def("__init__", [](soil::io::geotiff* geotiff, const soil::buffer& buffer, const soil::index& index){
  new (geotiff) soil::io::geotiff(buffer, index);
});

geotiff.def("meta", &soil::io::geotiff::meta);
geotiff.def("read", &soil::io::geotiff::read);
geotiff.def("write", &soil::io::geotiff::write);

geotiff.def_prop_ro("min", [](soil::io::geotiff& geotiff){
  return geotiff.min();
});

geotiff.def_prop_ro("max", [](soil::io::geotiff& geotiff){
  return geotiff.max();
});

geotiff.def_prop_ro("scale", [](soil::io::geotiff& geotiff){
  return geotiff.scale();
});

auto geotiff_meta = nb::class_<soil::io::geotiff::meta_t>(module, "geotiff_meta");

geotiff.def("get_meta", &soil::io::geotiff::get_meta);
geotiff.def("set_meta", &soil::io::geotiff::set_meta);

geotiff.def("unsetnan", &soil::io::geotiff::unsetNaN);

geotiff_meta.def_prop_ro("metadata", [](soil::io::geotiff::meta_t& m){
  return m.gdal_metadata;
});

geotiff_meta.def_prop_ro("nodata", [](soil::io::geotiff::meta_t& m){
  return m.gdal_nodata;
});

}

#endif