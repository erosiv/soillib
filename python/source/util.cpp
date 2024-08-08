#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
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

template<typename T>
void bind_array_t(py::module& module, const char* name){

using array_t = soil::array_t<T>;
auto array = py::class_<array_t>(module, name, py::buffer_protocol());

array.def("elem", &array_t::elem);
array.def("size", &array_t::size);
array.def("zero", &array_t::zero);
array.def("fill", &array_t::fill);

array.def_property_readonly("type", &array_t::type);
array.def_property_readonly("shape", &array_t::shape);

array.def("reshape", &array_t::reshape);

array.def("__getitem__", [](const array_t& array, const size_t index){
  return array[index];
});

array.def("__setitem__", [](array_t& array, const size_t index, const T& value){
  array[index] = value;
});

array.def("__setitem__", [](array_t& array, const py::tuple& tup, const T& value){

  std::vector<size_t> v;
  for(auto& d: tup)
    v.push_back(d.cast<size_t>());
  
  const size_t index = array.shape().flat(&v[0], v.size());
  array[index] = value;

});

array.def_buffer([](array_t& array) -> py::buffer_info {

  std::vector<py::ssize_t> shape;
  std::vector<py::ssize_t> strides;

  const size_t dims = array.shape().dims();

  for(size_t d = 0; d < dims; ++d){
    shape.push_back(array.shape()[d]);
  }

  size_t s = sizeof(T) * array.elem();
  for(size_t d = 0; d < dims; ++d){
    s = s / array.shape()[d];
    strides.push_back(s);
  }
  
  return py::buffer_info(
    array.data(),
    sizeof(T),
    py::format_descriptor<T>::format(),
    dims,
    shape,
    strides
  );

});

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
shape.def(py::init<const std::vector<size_t>&>());

shape.def("dims", &soil::shape::dims);
shape.def("elem", &soil::shape::elem);
shape.def("iter", &soil::shape::iter);
shape.def("flat", [](const soil::shape& shape, const std::vector<size_t>& v){
  return shape.flat(&v[0], v.size());
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

bind_array_t<int>(module, "array_int");
bind_array_t<float>(module, "array_float");
bind_array_t<double>(module, "array_double");
bind_array_t<soil::fvec2>(module, "array_fvec2");
bind_array_t<soil::fvec3>(module, "array_fvec3");

module.def("array", [](std::string type, std::vector<size_t> v) -> soil::array {
  soil::shape shape(v);
  if(type == "int")     return soil::array_t<int>(shape);
  if(type == "float")   return soil::array_t<float>(shape);
  if(type == "double")  return soil::array_t<double>(shape);
  throw std::invalid_argument("invalid type argument");
});

}

#endif