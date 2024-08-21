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

auto index = nb::class_<soil::index>(module, "index");
index.def(nb::init<const soil::vec2>());

index.def("dims", &soil::index::dims);
index.def("elem", &soil::index::elem);
index.def("iter", &soil::index::iter);

//! \todo replace this template with a selector
index.def("flatten", &soil::index::flatten<1>);
index.def("flatten", &soil::index::flatten<2>);
index.def("flatten", &soil::index::flatten<3>);
index.def("flatten", &soil::index::flatten<4>);

index.def("oob", &soil::index::oob<1>);
index.def("oob", &soil::index::oob<2>);
index.def("oob", &soil::index::oob<3>);
index.def("oob", &soil::index::oob<4>);

}

#endif