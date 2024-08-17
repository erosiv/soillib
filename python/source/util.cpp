#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "glm.hpp"
namespace py = pybind11;

#include <soillib/util/timer.hpp>
#include <soillib/util/types.hpp>
#include <soillib/util/shape.hpp>
#include <soillib/util/array.hpp>

//
//
//

template<typename T>
void bind_yield_t(py::module& module, const char* name){

using yield_t = soil::yield<T>;
auto yield = py::class_<yield_t>(module, name);

yield.def("__iter__", [](soil::yield<T>& iter){
  return py::make_iterator(iter.begin(), iter.end());
}, py::keep_alive<0, 1>());

}

//
//
//

//! General Util Binding Function
void bind_util(py::module& module){

//
// Timer Type Binding
//

auto timer = py::class_<soil::timer>(module, "timer");
timer.def(py::init<>());

timer.def("__enter__", [](soil::timer& timer){
  timer.start();
});

timer.def("__exit__", [](soil::timer& timer,
  const std::optional<pybind11::type>& exc_type,
  const std::optional<pybind11::object>& exc_value,
  const std::optional<pybind11::object>& traceback
){
  timer.stop();
  std::cout<<"Execution Time: "<<timer.count()<<" ms"<<std::endl;
});

//
// Yield Type Binding
//

bind_yield_t<soil::shape_t<1>::arr_t>(module, "yield_shape_t_arr_1");
bind_yield_t<soil::shape_t<2>::arr_t>(module, "yield_shape_t_arr_2");
bind_yield_t<soil::shape_t<3>::arr_t>(module, "yield_shape_t_arr_3");

//
// Shape Type Binding
//

auto shape = py::class_<soil::shape>(module, "shape");
shape.def(py::init<const std::vector<int>&>());

shape.def("dims", &soil::shape::dims);
shape.def("elem", &soil::shape::elem);
shape.def("iter", &soil::shape::iter);
shape.def("flat", [](const soil::shape& shape, const std::vector<int>& v){
  if(v.size() == 1) return shape.flat<1>({v[0]});
  if(v.size() == 2) return shape.flat<2>({v[0], v[1]});
  if(v.size() == 3) return shape.flat<3>({v[0], v[1], v[2]});
  throw std::invalid_argument("invalid size");
  //return shape.flat(&v[0], v.size());
});

shape.def("oob", [](const soil::shape& shape, const glm::ivec2 pos){
  return shape.oob(pos);
});

shape.def("__getitem__", &soil::shape::operator[]);

shape.def("__repr__", [](const soil::shape& shape){
  std::string str = "shape(" + std::to_string(shape.dims()) +")";
  str += "[";
  for(size_t d = 0; d < shape.dims(); d++)
    str += std::to_string(shape[d]) + ",";
  str += "]";
  return str;
});

//
// Array Type Binding
//

auto array = py::class_<soil::array>(module, "array", py::buffer_protocol());
array.def(py::init<const std::string, const soil::shape&>());
array.def(py::init<>([](const std::string type, const std::vector<int>& v){
  auto shape = soil::shape(v);
  return soil::array(type, shape);
}));

array.def("elem", &soil::array::elem);
array.def("size", &soil::array::size);
array.def("zero", &soil::array::zero);
array.def("fill", &soil::array::fill<int>);
array.def("fill", &soil::array::fill<float>);
array.def("fill", &soil::array::fill<double>);

array.def_property_readonly("type", &soil::array::type);
array.def_property_readonly("shape", &soil::array::shape);

array.def("reshape", &soil::array::reshape);

array.def_buffer([](soil::array& array) -> py::buffer_info {

  std::vector<py::ssize_t> shape;
  std::vector<py::ssize_t> strides;

  const size_t dims = array.shape().dims();

  for(size_t d = 0; d < dims; ++d){
    shape.push_back(array.shape()[d]);
  }

  size_t s = array.size();
  for(size_t d = 0; d < dims; ++d){
    s = s / array.shape()[d];
    strides.push_back(s);
  }
  
  std::string format;
  if(array.type() == "int") format = py::format_descriptor<int>::format();
  if(array.type() == "float") format = py::format_descriptor<float>::format();
  if(array.type() == "double") format = py::format_descriptor<double>::format();
  if(array.type() == "vec2") format = py::format_descriptor<soil::vec2>::format();
  if(array.type() == "vec3") format = py::format_descriptor<soil::vec3>::format();

  return py::buffer_info(
    array.data(),
    array.size() / array.elem(),
    format,
    dims,
    shape,
    strides
  );
});

array.def("__getitem__", &soil::array::operator[]);
array.def("__setitem__", &soil::array::set<int>);
array.def("__setitem__", &soil::array::set<float>);
array.def("__setitem__", &soil::array::set<double>);

array.def("__setitem__", [](soil::array& array, const py::tuple& tup, const py::object value){
  size_t index;
  if(tup.size() == 1) index = array.shape().template flat<1>({tup[0].cast<int>()});
  if(tup.size() == 2) index = array.shape().template flat<2>({tup[0].cast<int>(), tup[1].cast<int>()});
  if(tup.size() == 3) index = array.shape().template flat<3>({tup[0].cast<int>(), tup[1].cast<int>(), tup[2].cast<int>()});

  if(array.type() == "int") array.set<int>(index, value.cast<int>());
  if(array.type() == "float") array.set<float>(index, value.cast<float>());
  if(array.type() == "double") array.set<double>(index, value.cast<double>());
});

}

#endif