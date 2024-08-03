#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <variant>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <soillib/util/shape.hpp>
#include <soillib/util/array.hpp>

//! Templated Buffer-Type Binding
template<typename T>
void bind_array_t(py::module& module, const char* name){

using buf_t = soil::array_t<T>;
auto buffer = py::class_<buf_t, soil::array_b>(module, name);

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
  //array.reshape((std::vector<long int>)buf.shape());
  return array;
});

}

template<typename T>
void bind_yield_t(py::module& module, const char* name){

using yield_t = soil::yield<T>;
auto yield = py::class_<yield_t>(module, name);

yield.def("__iter__", [](soil::yield<T>& iter){
  return py::make_iterator(iter.begin(), iter.end());
}, py::keep_alive<0, 1>());

}

//! General Util Binding Function
void bind_util(py::module& module){

//
// Yield Type Binding
//

bind_yield_t<size_t>(module, "yield_size_t");

//
// Shape Type Binding
//

auto shape = py::class_<soil::shape>(module, "shape");

//bind_shape_t<1>(module, "shape_1");
//bind_shape_t<2>(module, "shape_2");
//bind_shape_t<3>(module, "shape_3");

shape.def("elem", &soil::shape::elem);

shape.def("__repr__", [](soil::shape& shape){
  std::string str = "shape(" + std::to_string(shape.dims()) +")";
  str += "[";
  for(size_t d = 0; d < shape.dims(); d++)
    str += std::to_string(shape[d]) + ",";
  str += "]";
  return str;
});

shape.def("iter", [](soil::shape& shape){
  return shape.iter();
}, py::keep_alive<0, 1>());

shape.def(py::init<std::vector<size_t>>());

/*

shape.def(py::init([](std::vector<size_t> v) -> soil::shape* {
  if(v.size() == 0) throw std::invalid_argument("vector can't have size 0");
  if(v.size() == 1) return new soil::shape_t<1>({v[0]});
  if(v.size() == 2) return new soil::shape_t<2>({v[0], v[1]});
  if(v.size() == 3) return new soil::shape_t<3>({v[0], v[1], v[2]});
throw std::invalid_argument("vector has invalid size");
}));
*/

//
// Buffer Type Binding
//

auto array = py::class_<soil::array>(module, "array");
auto array_b = py::class_<soil::array_b>(module, "array_b");

array_b.def("elem", &soil::array_b::elem);
array_b.def("size", &soil::array_b::size);
array_b.def("shape", &soil::array_b::shape);

bind_array_t<int>(module, "array_int");
bind_array_t<float>(module, "array_float");
bind_array_t<double>(module, "array_double");

array.def(py::init<>([](std::string type, std::vector<size_t> vec){
  return soil::array(type, vec);
}));

array.def("size", &soil::array::size);
array.def("elem", &soil::array::elem);
array.def("shape", [](soil::array& buf){
  auto s = buf.shape();
  return s;
}, py::return_value_policy::reference);

using buf_v = std::variant<
  soil::array_t<int>, 
  soil::array_t<float>,
  soil::array_t<double>
>;

array.def("to", [](soil::array& buf, std::string type) -> buf_v {
  if(type == "int") return buf_v{buf.as<int>()};
  if(type == "float") return buf_v{buf.as<float>()};
  if(type == "double") return buf_v{buf.as<double>()};
  throw std::invalid_argument("invalid argument for type");
});

}

#endif