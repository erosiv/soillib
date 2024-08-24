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
#include <soillib/core/types.hpp>
#include <soillib/core/index.hpp>
#include <soillib/core/buffer.hpp>

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

nb::enum_<soil::timer::duration>(module, "duration")
  .value("s",   soil::timer::duration::SECONDS)
  .value("ms",  soil::timer::duration::MILLISECONDS)
  .value("us",  soil::timer::duration::MICROSECONDS)
  .value("ns",  soil::timer::duration::NANOSECONDS)
  .export_values();

auto timer = nb::class_<soil::timer>(module, "timer");
timer.def(nb::init<const soil::timer::duration>());
timer.def(nb::init<>());  // default: milliseconds

timer.def("__enter__", [](soil::timer& timer){
  timer.start();
});

timer.def("__exit__", [](soil::timer& timer,
   std::optional<nb::handle>,
   std::optional<nb::object>,
   std::optional<nb::object>
){
  timer.stop();
  std::cout<<"Execution Time: "<<timer.count()<<std::endl;
}, nb::arg().none(), nb::arg().none(), nb::arg().none());

//
// Yield Type Binding
//

bind_yield_t<soil::flat_t<1>::vec_t>(module, "yield_shape_t_arr_1");
bind_yield_t<soil::flat_t<2>::vec_t>(module, "yield_shape_t_arr_2");
bind_yield_t<soil::flat_t<3>::vec_t>(module, "yield_shape_t_arr_3");
bind_yield_t<soil::flat_t<4>::vec_t>(module, "yield_shape_t_arr_4");

//
// Array Type Binding
//

auto buffer = nb::class_<soil::buffer>(module, "buffer");
buffer.def(nb::init<>());
buffer.def(nb::init<const soil::dtype, const size_t>());

buffer.def_prop_ro("type", &soil::buffer::type);

buffer.def("elem", &soil::buffer::elem);
buffer.def("size", &soil::buffer::size);

buffer.def("zero", [](soil::buffer& buffer){
  return soil::select(buffer.type(), [&buffer]<typename S>(){
    auto buffer_t = buffer.as<S>();
    for(size_t i = 0; i < buffer_t.elem(); ++i)
      buffer_t[i] = S{0};
    return buffer;
  });
});

buffer.def("fill", [](soil::buffer& buffer, const nb::object value){
  return soil::select(buffer.type(), [&buffer, &value]<typename S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    for(size_t i = 0; i < buffer_t.elem(); ++i)
      buffer_t[i] = value_t;
    return buffer;
  });
});

buffer.def("__getitem__", [](const soil::buffer& buffer, const size_t index) -> nb::object {
  return soil::select(buffer.type(), [&buffer, index]<typename S>() -> nb::object {
    S value = buffer.as<S>().operator[](index);
    return nb::cast<S>(std::move(value));
  });
});

buffer.def("__setitem__", [](soil::buffer& buffer, const size_t index, const nb::object value){
  soil::select(buffer.type(), [&buffer, index, &value]<typename S>(){
      buffer.as<S>()[index] = nb::cast<S>(value);
  });
});

//! \todo clean this up once the method for converting vector types is figured out.

buffer.def("numpy", [](soil::buffer& buffer) -> nb::object {
  return soil::select(buffer.type(), [&buffer]<typename S>() -> nb::object {
    if constexpr(std::same_as<S, int>){
      size_t shape[1]{buffer.elem()};
      nb::ndarray<nb::numpy, int, nb::ndim<1>> array(
        buffer.data(),
        1,
        shape,
        nb::handle()
      );
      return nb::cast(std::move(array));
    } else if constexpr(std::same_as<S, float>){
      size_t shape[1]{buffer.elem()};
      nb::ndarray<nb::numpy, float, nb::ndim<1>> array(
        buffer.data(),
        1,
        shape,
        nb::handle()
      );
      return nb::cast(std::move(array));
    } else if constexpr(std::same_as<S, double>){
      size_t shape[1]{buffer.elem()};
      nb::ndarray<nb::numpy, double, nb::ndim<1>> array(
        buffer.data(),
        1,
        shape,
        nb::handle()
      );
      return nb::cast(std::move(array));
    } else if constexpr(std::same_as<S, soil::vec2>){
      size_t shape[1]{2*buffer.elem()};
      nb::ndarray<nb::numpy, float, nb::ndim<1>> array(
        buffer.data(),
        1,
        shape,
        nb::handle()
      );
      return nb::cast(std::move(array));
    } else if constexpr(std::same_as<S, soil::vec3>){
      size_t shape[1]{3*buffer.elem()};
      nb::ndarray<nb::numpy, float, nb::ndim<1>> array(
        buffer.data(),
        1,
        shape,
        nb::handle()
      );
      return nb::cast(std::move(array));
    }
    throw std::invalid_argument("can't convert this buffer to numpy");
  });
});

}

#endif