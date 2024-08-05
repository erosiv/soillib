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

template<size_t N>
void bind_shape_t(py::module& module, const char* name){

using shape_t = soil::shape_t<N>;
auto shape = py::class_<shape_t>(module, name);

shape.def("dims", &shape_t::dims);
shape.def("elem", &shape_t::elem);
shape.def("flat", &shape_t::flat);
shape.def("iter", &shape_t::iter, py::keep_alive<0, 1>());

shape.def("__getitem__", [](const shape_t& shape, const size_t index){
  return shape[index];
});

shape.def("__repr__", [](const shape_t& shape){
  std::string str = "shape(" + std::to_string(shape.dims()) +")";
  str += "[";
  for(size_t d = 0; d < shape.dims(); d++)
    str += std::to_string(shape[d]) + ",";
  str += "]";
  return str;
});

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

array.def_buffer([](array_t& array) -> py::buffer_info {

  std::vector<py::ssize_t> shape;
  std::vector<py::ssize_t> strides;

  const size_t dims = std::visit([](auto&& args){
    return args.dims();    
  }, array.shape());

  auto get_d = [](const soil::shape& shape, const size_t d){
    return std::visit([d](auto&& args){
      return args[d];
    }, shape);
  };

  for(size_t d = 0; d < dims; ++d){
    shape.push_back(get_d(array.shape(), d));
  }

  size_t s = sizeof(T) * array.elem();
  for(size_t d = 0; d < dims; ++d){
    s = s / get_d(array.shape(), d);
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

bind_shape_t<1>(module, "shape_1");
bind_shape_t<2>(module, "shape_2");
bind_shape_t<3>(module, "shape_3");

auto do_test = [](std::vector<size_t> v) -> soil::shape {
  if(v.size() == 1) return soil::shape_t<1>{v};
  if(v.size() == 2) return soil::shape_t<2>{v};
  if(v.size() == 3) return soil::shape_t<3>{v};
  throw std::invalid_argument("too many dimensions?");
};

module.def("shape", do_test);

//
// Array Type Binding
//

bind_array_t<int>(module, "array_int");
bind_array_t<float>(module, "array_float");
bind_array_t<double>(module, "array_double");

module.def("array", [&do_test](std::string type, std::vector<size_t> v) -> soil::array { 
  soil::shape shape = do_test(v);
  if(type == "int")     return soil::array_t<int>(shape);
  if(type == "float")   return soil::array_t<float>(shape);
  if(type == "double")  return soil::array_t<double>(shape);
  throw std::invalid_argument("invalid type argument");
});


/*
// Wrapper-Class Implementation

auto array = py::class_<soil::array>(module, "array", );

array.def(py::init<>([](std::string type, const soil::shape& shape){
  return soil::array(type, shape);
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

array.def("__setitem__", [](soil::array& source, py::tuple& tup, int value){
  size_t index;
  if(tup.size() == 1) index = source.shape().flat<1>({tup[0].cast<size_t>()});
  if(tup.size() == 2) index = source.shape().flat<2>({tup[0].cast<size_t>(), tup[1].cast<size_t>()});
  if(tup.size() == 3) index = source.shape().flat<3>({tup[0].cast<size_t>(), tup[1].cast<size_t>(), tup[2].cast<size_t>()});
  source.set<int>(index, value);
});

array.def("__setitem__", [](soil::array& source, py::tuple& tup, float value){
  size_t index;
  if(tup.size() == 1) index = source.shape().flat<1>({tup[0].cast<size_t>()});
  if(tup.size() == 2) index = source.shape().flat<2>({tup[0].cast<size_t>(), tup[1].cast<size_t>()});
  if(tup.size() == 3) index = source.shape().flat<3>({tup[0].cast<size_t>(), tup[1].cast<size_t>(), tup[2].cast<size_t>()});
  source.set<float>(index, value);
});

array.def("__setitem__", [](soil::array& source, py::tuple& tup, double value){
  size_t index;
  if(tup.size() == 1) index = source.shape().flat<1>({tup[0].cast<size_t>()});
  if(tup.size() == 2) index = source.shape().flat<2>({tup[0].cast<size_t>(), tup[1].cast<size_t>()});
  if(tup.size() == 3) index = source.shape().flat<3>({tup[0].cast<size_t>(), tup[1].cast<size_t>(), tup[2].cast<size_t>()});
  source.set<double>(index, value);
});

array.def("__getitem__", [](soil::array& source, py::tuple& tup){
  if(tup.size() == 1) return source.get<1>({tup[0].cast<size_t>()});
  if(tup.size() == 2) return source.get<2>({tup[0].cast<size_t>(), tup[1].cast<size_t>()});
  if(tup.size() == 3) return source.get<3>({tup[0].cast<size_t>(), tup[1].cast<size_t>(), tup[2].cast<size_t>()});
  throw std::invalid_argument("invalid tuple size");
});

// array.def("__getitem__", [](soil::array& source, const soil::shape_t<1>::arr_t pos){
//   return source.get<1>(pos);
// });
// 
// array.def("__getitem__", [](soil::array& source, const soil::shape_t<2>::arr_t pos){
//   return source.get<2>(pos);
// });
// 
// array.def("__getitem__", [](soil::array& source, const soil::shape_t<3>::arr_t pos){
//   return source.get<3>(pos);
// });

array.def_buffer([](soil::array& array) -> py::buffer_info {
  if(array.type() == "int") return make_buffer<int>(array);
  if(array.type() == "float") return make_buffer<float>(array);
  if(array.type() == "double") return make_buffer<double>(array);
  throw std::invalid_argument("invalid argument for type");
});

*/


}

#endif