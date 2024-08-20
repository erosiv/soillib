#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>   //! \todo TRY TO ELIMINATE
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/optional.h>

#include <soillib/util/timer.hpp>
#include <soillib/util/types.hpp>
#include <soillib/util/shape.hpp>
#include <soillib/util/buffer.hpp>

#include "glm.hpp"

//
//
//

template<typename T>
void bind_yield_t(nb::module_& module, const char* name){

using yield_t = soil::yield<T>;
auto yield = nb::class_<yield_t>(module, name);

yield.def("__iter__", [](soil::yield<T>& iter){
  return nb::make_iterator(nb::type<soil::yield<T>>(), "iterator",
    iter.begin(), iter.end());
}, nb::keep_alive<0, 1>());

}

//
//
//

//! General Util Binding Function
void bind_util(nb::module_& module){

//
// Type Enumerator Binding
//

nb::enum_<soil::dtype>(module, "dtype")
  .value("int", soil::dtype::INT)
  .value("float32", soil::dtype::FLOAT32)
  .value("float64", soil::dtype::FLOAT64)
  .value("vec2", soil::dtype::VEC2)
  .export_values();

//
// Timer Type Binding
//

auto timer = nb::class_<soil::timer>(module, "timer");
timer.def(nb::init<>());

timer.def("__enter__", [](soil::timer& timer){
  timer.start();
});

timer.def("__exit__", [](soil::timer& timer,
   std::optional<nb::handle>,
   std::optional<nb::object>,
   std::optional<nb::object>
){
  timer.stop();
  std::cout<<"Execution Time: "<<timer.count()<<" ms"<<std::endl;
}, nb::arg().none(), nb::arg().none(), nb::arg().none());

//
// Yield Type Binding
//

bind_yield_t<soil::shape_t<1>::arr_t>(module, "yield_shape_t_arr_1");
bind_yield_t<soil::shape_t<2>::arr_t>(module, "yield_shape_t_arr_2");
bind_yield_t<soil::shape_t<3>::arr_t>(module, "yield_shape_t_arr_3");

//
// Shape Type Binding
//

auto shape = nb::class_<soil::shape>(module, "shape");
shape.def(nb::init<const std::vector<int>&>());

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

shape.def("flat", [](const soil::shape& shape, glm::vec2 pos){
  return shape.flat(pos);
});

shape.def("oob", [](const soil::shape& shape, const glm::ivec2 pos){
  return shape.oob(pos);
});

shape.def("oob", [](const soil::shape& shape, const glm::vec2 pos){
  return shape.oob(glm::ivec2(pos));
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

auto buffer = nb::class_<soil::buffer>(module, "buffer");
buffer.def(nb::init<>());
buffer.def(nb::init<const soil::dtype, const size_t>());

buffer.def("elem", &soil::buffer::elem);
buffer.def("size", &soil::buffer::size);
buffer.def("zero", &soil::buffer::zero);
buffer.def("fill", &soil::buffer::fill<int>);
buffer.def("fill", &soil::buffer::fill<float>);
buffer.def("fill", &soil::buffer::fill<double>);
buffer.def("fill", &soil::buffer::fill<soil::vec2>);

buffer.def_prop_ro("type", &soil::buffer::type);

buffer.def("__getitem__", [](const soil::buffer& buffer, const size_t index) -> nb::object {
  return soil::typeselect(buffer.type(), [&buffer, index]<typename S>() -> nb::object {
    S value = buffer.as<S>().operator[](index);
    return nb::cast<S>(std::move(value));
  });
});

buffer.def("__setitem__", [](soil::buffer& buffer, const size_t index, const nb::object value){
  soil::typeselect(buffer.type(), [&buffer, index, &value]<typename S>(){
      buffer.as<S>()[index] = nb::cast<S>(value);
  });
});

//! \todo clean this up once the method for converting vector types is figured out.

buffer.def("numpy", [](soil::buffer& buffer) -> nb::ndarray<nb::numpy, float, nb::ndim<1>> {
  return soil::typeselect(buffer.type(), [&buffer]<typename S>() -> nb::ndarray<nb::numpy, float, nb::ndim<1>> {
    if constexpr(std::same_as<S, float>){
      size_t shape[1]{buffer.elem()};
      return nb::ndarray<nb::numpy, float, nb::ndim<1>>(
        buffer.data(),
        1,
        shape,
        nb::handle()
      );
    } else if constexpr(std::same_as<S, soil::vec2>){
      size_t shape[1]{2*buffer.elem()};
      return nb::ndarray<nb::numpy, float, nb::ndim<1>>(
        buffer.data(),
        1,
        shape,
        nb::handle()
      );
    } else if constexpr(std::same_as<S, soil::vec3>){
      size_t shape[1]{3*buffer.elem()};
      return nb::ndarray<nb::numpy, float, nb::ndim<1>>(
        buffer.data(),
        1,
        shape,
        nb::handle()
      );
    }
    throw std::invalid_argument("can't convert this buffer to numpy");
  });
});

}

#endif