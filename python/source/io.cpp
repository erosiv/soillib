#include <soillib/soillib.hpp>

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <soillib/io/tiff.hpp>
#include <soillib/io/geotiff.hpp>

#include "glm.hpp"

void bind_io(nb::module_& module){

//! TIFF Datatype

auto tiff = nb::class_<soil::io::tiff>(module, "tiff");

tiff.def(nb::init<>());
tiff.def(nb::init<const char*>());
tiff.def("__init__", [](soil::io::tiff* tiff, const silt::tensor& tensor){
  new (tiff) soil::io::tiff(tensor);
});

tiff.def("peek", &soil::io::tiff::peek);
tiff.def("read", &soil::io::tiff::read);
tiff.def("write", &soil::io::tiff::write);

tiff.def_prop_ro("width", &soil::io::tiff::width);
tiff.def_prop_ro("height", &soil::io::tiff::height);
tiff.def_prop_ro("samples", &soil::io::tiff::samples);
tiff.def_prop_ro("tensor", &soil::io::tiff::tensor);
tiff.def_prop_ro("shape", &soil::io::tiff::shape);

//! GeoTIFF Datatype

auto geotiff = nb::class_<soil::io::geotiff, soil::io::tiff>(module, "geotiff");

geotiff.def(nb::init<>());
geotiff.def(nb::init<const char*>());
geotiff.def("__init__", [](soil::io::geotiff* geotiff, const silt::tensor& tensor){
  new (geotiff) soil::io::geotiff(tensor);
});

geotiff.def("peek", &soil::io::geotiff::peek);
geotiff.def("read", &soil::io::geotiff::read);
geotiff.def("write", &soil::io::geotiff::write);

geotiff.def_rw("meta", &soil::io::geotiff::_meta);

geotiff.def("unsetnan", &soil::io::geotiff::unsetNaN);

geotiff.def_prop_ro("min", [](soil::io::geotiff& geotiff){
  return geotiff.min();
});

geotiff.def_prop_ro("max", [](soil::io::geotiff& geotiff){
  return geotiff.max();
});

geotiff.def_prop_ro("scale", [](soil::io::geotiff& geotiff){
  return geotiff.scale();
});

//
// Geotiff Metadata
//

auto geotiff_meta = nb::class_<soil::io::geotiff::meta_t>(module, "geotiff_meta");

geotiff_meta.def_ro("filename", &soil::io::geotiff::meta_t::filename);
geotiff_meta.def_rw("width", &soil::io::geotiff::meta_t::width);
geotiff_meta.def_rw("height", &soil::io::geotiff::meta_t::height);
geotiff_meta.def_rw("bits", &soil::io::geotiff::meta_t::bits);
geotiff_meta.def_rw("samples", &soil::io::geotiff::meta_t::samples);

geotiff_meta.def_prop_rw("scale", [](const soil::io::geotiff::meta_t& geotiff_meta){
  return glm::vec3(geotiff_meta.scale[0], geotiff_meta.scale[1], geotiff_meta.scale[2]);
}, [](soil::io::geotiff::meta_t& geotiff_meta, const std::vector<double> scale){
  geotiff_meta.scale = scale;
});

geotiff_meta.def_rw("scale", &soil::io::geotiff::meta_t::scale);
geotiff_meta.def_rw("coords", &soil::io::geotiff::meta_t::coords);

geotiff_meta.def_rw("gdal_ascii", &soil::io::geotiff::meta_t::geoasciiparams);
geotiff_meta.def_rw("gdal_metadata", &soil::io::geotiff::meta_t::gdal_metadata);
geotiff_meta.def_rw("gdal_nodata", &soil::io::geotiff::meta_t::gdal_nodata);

// min is the SOUTH-WEST corner in world space, i.e. tensor cell (0, 0). The
// format stores the NORTH-west tie point, so the setter converts; assign
// `scale` before `min` so the conversion has the cell size it needs.
geotiff_meta.def_prop_rw("min", [](soil::io::geotiff::meta_t& geotiff_meta){
  return geotiff_meta.min();
}, [](soil::io::geotiff::meta_t& geotiff_meta, const std::vector<double> min){
  geotiff_meta.set_min(glm::vec2(min[0], min[1]));
});

geotiff_meta.def_prop_ro("max", [](soil::io::geotiff::meta_t& geotiff_meta){
  return geotiff_meta.max();
});

}
