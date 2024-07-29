#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <variant>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <soillib/util/new/buf.hpp>
#include <soillib/util/new/tiff.hpp>
#include <soillib/util/new/geotiff.hpp>

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

buffer.def("to", [](soil::buffer& buf, std::string type) -> buf_v {
  if(type == "int") return buf_v{buf.as<int>()};
  if(type == "float") return buf_v{buf.as<float>()};
  if(type == "double") return buf_v{buf.as<double>()};
  throw std::invalid_argument("invalid argument for type");
});

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