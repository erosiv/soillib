#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <variant>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#define SHAPE_IMPL

#include <soillib/util/new/shape.hpp>
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

//! Templated Dimension-Type Binding
template<size_t N>
void bind_shape_t(py::module& module, const char* name){

// using dim_t = soil::dim_t<N>;
// auto dim = py::class_<dim_t>(module, name);

using shape_t = soil::shape_t<N>;
auto shape = py::class_<shape_t, soil::shape>(module, name);

shape.def(py::init<const size_t>());
shape.def("elem", &shape_t::elem);
shape.def("flat", &shape_t::flat);
shape.def("__getitem__", [](shape_t& shape, const size_t index) -> size_t {
  return shape[index];
});

shape.def("__iter__", [](shape_t& shape){
  return py::make_iterator(shape.begin(), shape.end());
}, py::keep_alive<0, 1>());

shape.def("__repr__", [](shape_t& shape){
  return "shape(" + std::to_string(shape.n_dim) +")";
});

}

//! General Util Binding Function
void bind_util(py::module& module){

// Shape Type Binding

auto shape = py::class_<soil::shape>(module, "shape");

bind_shape_t<1>(module, "shape_1");
bind_shape_t<2>(module, "shape_2");
bind_shape_t<3>(module, "shape_3");

shape.def("elem", &soil::shape::elem);

shape.def(py::init([](std::vector<size_t> v){
  if(v.size() == 0) throw std::invalid_argument("vector can't have size 0");
  if(v.size() == 1) return soil::shape::make(v[0]);
  if(v.size() == 2) return soil::shape::make(v[0], v[1]);
  if(v.size() == 3) return soil::shape::make(v[0], v[1], v[2]);
throw std::invalid_argument("vector has invalid size");
}));

// Buffer Type Binding

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

}

#endif