#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include <soillib/util/timer.hpp>
#include <soillib/core/types.hpp>
#include <soillib/core/index.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/util/error.hpp>

#include <soillib/op/math.hpp>

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
  .value("vec3", soil::dtype::VEC3)
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
//  std::cout<<"Execution Time: "<<timer.count()<<std::endl;
}, nb::arg().none(), nb::arg().none(), nb::arg().none());

timer.def_prop_ro("count", [](const soil::timer& timer){
  return timer.count();
});

//
// Yield Type Binding
//

bind_yield_t<soil::flat_t<1>::vec_t>(module, "yield_shape_t_arr_1");
bind_yield_t<soil::flat_t<2>::vec_t>(module, "yield_shape_t_arr_2");
bind_yield_t<soil::flat_t<3>::vec_t>(module, "yield_shape_t_arr_3");
bind_yield_t<soil::flat_t<4>::vec_t>(module, "yield_shape_t_arr_4");

}

#endif