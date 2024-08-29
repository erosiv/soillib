#ifndef SOILLIB_PYTHON_UTIL
#define SOILLIB_PYTHON_UTIL

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>   //! \todo TRY TO ELIMINATE
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

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
void bind_index(nb::module_& module){

nb::enum_<soil::dindex>(module, "dindex")
  .value("flat1", soil::dindex::FLAT1)
  .value("flat2", soil::dindex::FLAT2)
  .value("flat3", soil::dindex::FLAT3)
  .value("flat4", soil::dindex::FLAT4)
  .value("quad", soil::dindex::QUAD)
  .export_values();

//
// Shape Type Binding
//

auto index = nb::class_<soil::index>(module, "index");

index.def(nb::init<const soil::ivec1>());
index.def(nb::init<const soil::ivec2>());
index.def(nb::init<const soil::ivec3>());
index.def(nb::init<const soil::ivec4>());

// VERY SPECIAL CONSTRUCTOR!

index.def("__init__", [](soil::index* index, const std::vector<std::tuple<soil::index::vec_t<2>, soil::index::vec_t<2>>> data){
  new (index) soil::index(data);
});

index.def_prop_ro("type", &soil::index::type);
index.def("dims", &soil::index::dims);
index.def("elem", &soil::index::elem);

index.def("__getitem__", &soil::index::operator[]);

index.def("min", [](soil::index& index) -> nb::object {
  return soil::select(index.type(), [&index]<typename T>() -> nb::object {
    auto min = index.as<T>().min();
    return nb::cast(std::move(min));
  });
});

index.def("max", [](soil::index& index) -> nb::object {
  return soil::select(index.type(), [&index]<typename T>() -> nb::object {
    auto max = index.as<T>().max();
    return nb::cast(std::move(max));
  });
});

index.def("ext", [](soil::index& index) -> nb::object {
  return soil::select(index.type(), [&index]<typename T>() -> nb::object {
    auto ext = index.as<T>().ext();
    return nb::cast(std::move(ext));
  });
});

index.def("iter", [](soil::index& index) -> nb::object {
  return soil::select(index.type(), [&index]<typename T>() -> nb::object {
    auto iter = index.as<T>().iter();
    return nb::cast(std::move(iter));
  });
});

// note: necessary for floating point positions!
// note: has to be defined first for higher priority
index.def("oob", [](soil::index& index, soil::vec2 vec){
  return index.oob<2>(vec);
});

index.def("oob", [](soil::index& index, nb::object& object) -> bool {
  return soil::select(index.type(), [&index, &object]<typename T>() -> bool {
    auto value = nb::cast<typename T::vec_t>(object);
    return index.as<T>().oob(value);
  });
});

index.def("flatten", [](soil::index& index, nb::object& object) -> size_t {
  return soil::select(index.type(), [&index, &object]<typename T>() -> size_t {
    auto value = nb::cast<typename T::vec_t>(object);
    return index.as<T>().flatten(value);
  });
});

index.def("unflatten", [](soil::index& index, const size_t s) -> nb::object {
  return soil::select(index.type(), [&index, s]<typename T>() -> nb::object {
    auto value = index.as<T>().unflatten(s);
    return nb::cast(value);
  });
});

}

#endif