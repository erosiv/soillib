#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <variant>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <soillib/util/new/buf.hpp>

//! Templated Buffer-Type Binding
template<typename T>
void bind_buf_t(py::module& module, const char* name){

using buf_t = soil::buf_t<T>;
auto buffer = py::class_<buf_t, soil::buffer>(module, name);

buffer.def("zero", &buf_t::zero);
buffer.def("fill", &buf_t::fill);

buffer.def("__getitem__", [](buf_t& buf, const size_t index) -> T {
  return buf[index];
});

buffer.def("__setitem__", [](buf_t& buf, const size_t index, T value){
  buf[index] = value;
});

buffer.def("numpy", [](buf_t& buf){
  py::array_t<T> array(buf.elem());
  py::buffer_info info = array.request();
  std::memcpy(info.ptr, buf.data(), buf.size());
  return array;
});

}

//! General Util Binding Function
void bind_util(py::module& module){

auto buffer = py::class_<soil::buffer>(module, "buffer");

bind_buf_t<int>(module, "buffer_int");
bind_buf_t<float>(module, "buffer_float");
bind_buf_t<double>(module, "buffer_double");

buffer.def(py::init<>([](std::string type, size_t size){
  return soil::buffer::make(type, size);
}));

buffer.def("size", &soil::buffer::size);
buffer.def("elem", &soil::buffer::elem);

using buf_v = std::variant<
  soil::buf_t<int>, 
  soil::buf_t<float>,
  soil::buf_t<double>
>;

buffer.def("ast", [](soil::buffer& buf, std::string type) {
  return buf.as<float>();
  /*
  if(type == "int") return buf_v(buf.as<int>());
  // if(type == "float") return {buf.as<float>()};
  // if(type == "double") return {buf.as<double>()};
  */
  throw std::invalid_argument("invalid argument for type");
});

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
*/

}

#endif