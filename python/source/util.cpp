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

template<typename T>
py::buffer_info make_buffer(soil::array& source){

  std::vector<py::ssize_t> shape;
  std::vector<py::ssize_t> strides;

  for(size_t d = 0; d < source.shape().dims(); ++d){
    shape.push_back(source.shape()[d]);
  }

  size_t s = sizeof(T) * source.shape().elem();
  for(size_t d = 0; d < source.shape().dims(); ++d){
    s = s / source.shape()[d];
    strides.push_back(s);
  }
  
  return py::buffer_info(
    source.data(),
    sizeof(T),
    py::format_descriptor<T>::format(),
    source.shape().dims(),
    shape,
    strides
  );
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
void bind_util(py::module& module){\

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

shape.def(py::init<std::vector<size_t>>());

shape.def("elem", &soil::shape::elem);
shape.def("dims", &soil::shape::dims);

shape.def("flat", &soil::shape::flat<1>);
shape.def("flat", &soil::shape::flat<2>);
shape.def("flat", &soil::shape::flat<3>);

using shape_iter_t = std::variant<
  soil::yield<soil::shape_t<1>::arr_t>,
  soil::yield<soil::shape_t<2>::arr_t>,
  soil::yield<soil::shape_t<3>::arr_t>
>;

shape.def("iter", [](soil::shape& shape) -> shape_iter_t {
  if(shape.dims() == 1) return shape.as<1>()->iter();
  if(shape.dims() == 2) return shape.as<2>()->iter();
  if(shape.dims() == 3) return shape.as<3>()->iter();
  throw std::invalid_argument("too many dimensions?");
}, py::keep_alive<0, 1>()); 

shape.def("__getitem__", [](soil::shape& shape, const size_t index){
  return shape[index];
});

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

auto array = py::class_<soil::array>(module, "array", py::buffer_protocol());

array.def(py::init<>([](std::string type, std::vector<size_t> vec){
  return soil::array(type, vec);
}));

array.def("size", &soil::array::size);
array.def("elem", &soil::array::elem);

array.def_property_readonly("type", &soil::array::type);
array.def_property_readonly("shape", &soil::array::shape);

array.def("zero", &soil::array::zero);

array.def("fill", &soil::array::fill<int>);
array.def("fill", &soil::array::fill<float>);
array.def("fill", &soil::array::fill<double>);

array.def("reshape", &soil::array::reshape);

array.def("__setitem__", &soil::array::set<int>);
array.def("__setitem__", &soil::array::set<float>);
array.def("__setitem__", &soil::array::set<double>);

using val_v = soil::multi;

array.def("__getitem__", [](soil::array& source, const size_t index){
  return source.get(index);
});

array.def("__getitem__", [](soil::array& source, const soil::shape_t<1>::arr_t pos){
  return source.get<1>(pos);
});

array.def("__getitem__", [](soil::array& source, const soil::shape_t<2>::arr_t pos){
  return source.get<2>(pos);
});

array.def("__getitem__", [](soil::array& source, const soil::shape_t<3>::arr_t pos){
  return source.get<3>(pos);
});

array.def_buffer([](soil::array& array) -> py::buffer_info {
  if(array.type() == "int") return make_buffer<int>(array);
  if(array.type() == "float") return make_buffer<float>(array);
  if(array.type() == "double") return make_buffer<double>(array);
  throw std::invalid_argument("invalid argument for type");
});

}

#endif