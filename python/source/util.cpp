#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

void bind_util(py::module& module){

/*

auto cam_orthogonal = py::class_<Tiny::cam::orthogonal>(module, "cam_orthogonal");
cam_orthogonal.def(py::init<glm::vec2, glm::vec2, float>());
cam_orthogonal.def("hook", &Tiny::cam::orthogonal::hook);
cam_orthogonal.def("update", &Tiny::cam::orthogonal::update);
cam_orthogonal.def("proj", &Tiny::cam::orthogonal::proj);

auto cam_orbit = py::class_<Tiny::cam::orbit>(module, "cam_orbit");
cam_orbit.def(py::init<glm::vec3, glm::vec3>());
cam_orbit.def("hook", &Tiny::cam::orbit::hook);
cam_orbit.def("update", &Tiny::cam::orbit::update);
cam_orbit.def("view", &Tiny::cam::orbit::view);

using cam_ortho_orbit_t = Tiny::camera<Tiny::cam::orthogonal, Tiny::cam::orbit>;
auto cam_ortho_orbit = py::class_<cam_ortho_orbit_t>(module, "cam_ortho_orbit");
cam_ortho_orbit.def("vp", &cam_ortho_orbit_t::vp);
cam_ortho_orbit.def("hook", &cam_ortho_orbit_t::hook);
cam_ortho_orbit.def("update", &cam_ortho_orbit_t::update);

module.def("camera", [](const Tiny::cam::orthogonal& ortho, const Tiny::cam::orbit& orbit){
  return Tiny::camera(ortho, orbit);
}, py::keep_alive<0, 1>(), py::keep_alive<0, 2>());

// Image Bindings

using png_dat_t = glm::tvec4<uint8_t>;
using png_t = Tiny::png<png_dat_t>;
auto png = py::class_<png_t>(module, "png");

png.def(py::init<>());
png.def(py::init<const char*>());
png.def(py::init<const size_t, const size_t>());
png.def(py::init<const glm::ivec2>());

png.def_property_readonly("width", [](const png_t& png){ return png.width(); });
png.def_property_readonly("height", [](const png_t& png){ return png.height(); });
png.def_property_readonly("data", [](const png_t& png){ return std::unique_ptr<const png_t::value_t>(png.data()); });

png.def("allocate", &png_t::allocate);
png.def("read", &png_t::read);
png.def("write", &png_t::write);
png.def("__getitem__", [](png_t& png, const glm::ivec2 pos){
  return png[pos];
});

png.def("numpy", [](const png_t& png){
  const size_t elem = png.width()*png.height()*4;
  py::array_t<uint8_t> array(elem);
  py::buffer_info info = array.request();
  std::memcpy(info.ptr, png.data(), png.size());
  return array;
});

*/

}

#endif