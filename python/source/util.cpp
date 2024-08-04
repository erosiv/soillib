#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <soillib/util/types.hpp>
#include <soillib/util/shape.hpp>
#include <soillib/util/array.hpp>

template<typename T>
py::array_t<T> make_numpy(soil::array& source){
  py::array_t<T> array(source.elem());
  py::buffer_info info = array.request();
  std::memcpy(info.ptr, source.data(), source.size());
  return array;
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

shape.def(py::init<std::vector<size_t>>());

shape.def("elem", &soil::shape::elem);
shape.def("dims", &soil::shape::dims);

shape.def("iter", [](soil::shape& shape){
  return shape.iter();
}, py::keep_alive<0, 1>());

shape.def("__repr__", [](soil::shape& shape){
  std::string str = "shape(" + std::to_string(shape.dims()) +")";
  str += "[";
  for(size_t d = 0; d < shape.dims(); d++)
    str += std::to_string(shape[d]) + ",";
  str += "]";
  return str;
});

//
// Buffer Type Binding
//

// Wrapper-Class Implementation

auto array = py::class_<soil::array>(module, "array");

array.def(py::init<>([](std::string type, std::vector<size_t> vec){
  return soil::array(type, vec);
}));

array.def("type", &soil::array::type);
array.def("size", &soil::array::size);
array.def("elem", &soil::array::elem);
array.def("shape", &soil::array::shape);

array.def("zero", &soil::array::zero);

array.def("fill", &soil::array::fill<int>);
array.def("fill", &soil::array::fill<float>);
array.def("fill", &soil::array::fill<double>);

array.def("__setitem__", &soil::array::set<int>);
array.def("__setitem__", &soil::array::set<float>);
array.def("__setitem__", &soil::array::set<double>);

using val_v = soil::multi;

array.def("__getitem__", [](soil::array& a, const size_t index) -> val_v {
  if(a.type() == "int") return a.as<int>()[index];
  if(a.type() == "float") return a.as<float>()[index];
  if(a.type() == "double") return a.as<double>()[index];
  throw std::invalid_argument("invalid argument for type");
});

using arr_v = soil::multi_t<py::array_t>;

array.def("numpy", [](soil::array& a) -> arr_v {
  if(a.type() == "int") return make_numpy<int>(a);
  if(a.type() == "float") return make_numpy<float>(a);
  if(a.type() == "double") return make_numpy<double>(a);
  throw std::invalid_argument("invalid argument for type");
});

}

#endif