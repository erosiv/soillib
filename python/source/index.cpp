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
#include <soillib/index/index.hpp>
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
void bind_index(nb::module_& module){

//
// Shape Type Binding
//

auto shape = nb::class_<soil::shape>(module, "shape");
shape.def(nb::init<const std::vector<int>&>());

shape.def("dims", &soil::shape::dims);
shape.def("elem", &soil::shape::elem);
shape.def("iter", &soil::shape::iter);

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

}

#endif