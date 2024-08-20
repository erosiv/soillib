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
#include <soillib/util/array.hpp>

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

template<typename T, size_t N, size_t K>
nb::ndarray<nb::numpy, T, nb::ndim<N>> make_numpy(soil::array& array){

  const size_t dims = array.shape().dims();
  size_t _shape[N];
  for(size_t d = 0; d < dims; ++d){
    _shape[d] = array.shape()[d];
  };

  if(N == 3){
    _shape[2] = K;
  }

  return nb::ndarray<nb::numpy, T, nb::ndim<N>>(
    array.data(),
    N,
    _shape,
    nb::handle()
  );
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

auto array = nb::class_<soil::array>(module, "array");
array.def(nb::init<>());
array.def(nb::init<const soil::dtype, const soil::shape&>());
array.def("__init__", [](soil::array* array, const soil::dtype type, const std::vector<int>& v){ 
  auto shape = soil::shape(v);
  new (array) soil::array(type, shape);
});

array.def("elem", &soil::array::elem);
array.def("size", &soil::array::size);
array.def("zero", &soil::array::zero);
array.def("fill", &soil::array::fill<int>);
array.def("fill", &soil::array::fill<float>);
array.def("fill", &soil::array::fill<double>);
array.def("fill", &soil::array::fill<soil::vec2>);

array.def_prop_ro("type", &soil::array::type);
array.def_prop_ro("shape", &soil::array::shape);

array.def("__getitem__", [](const soil::array& array, const size_t index) -> nb::object {
  return soil::typeselect(array.type(), [&array, index]<typename S>() -> nb::object {
    S value = array.as<S>().operator[](index);
    return nb::cast<S>(std::move(value));
  });
});

array.def("__setitem__", [](soil::array& array, const size_t index, const nb::object value){
  soil::typeselect(array.type(), [&array, index, &value]<typename S>(){
      array.as<S>()[index] = nb::cast<S>(value);
  });
});

//! \todo replace the following with a simple flattening tuple caster for the index.

array.def("__setitem__", [](soil::array& array, const nb::tuple& tup, const nb::object value){

  size_t index;
  if(tup.size() == 1) index = array.shape().template flat<1>({nb::cast<int>(tup[0])});
  if(tup.size() == 2) index = array.shape().template flat<2>({nb::cast<int>(tup[0]), nb::cast<int>(tup[1])});
  if(tup.size() == 3) index = array.shape().template flat<3>({nb::cast<int>(tup[0]), nb::cast<int>(tup[1]), nb::cast<int>(tup[2])});

  soil::typeselect(array.type(), [&array, index, &value]<typename S>(){
      array.as<S>()[index] = nb::cast<S>(value);
  });

});

using test = std::variant<
  nb::ndarray<nb::numpy, float, nb::ndim<2>>,
  nb::ndarray<nb::numpy, float, nb::ndim<3>>
>;

array.def("numpy", [](soil::array& array) -> test {
  if(array.type() == soil::FLOAT32) return make_numpy<float, 2, 1>(array);
  if(array.type() == soil::VEC2)  return make_numpy<float, 3, 2>(array);
  if(array.type() == soil::VEC3) return make_numpy<float, 3, 3>(array);
  throw std::invalid_argument("I don't know how to make this into numpy!");
});









array.def("track_float", [](soil::array& lhs, soil::array& rhs, const float lrate){

  auto lhs_t = lhs.as<float>();
  auto rhs_t = rhs.as<float>();

  for(size_t i = 0; i < lhs.shape().elem(); ++i){
    float lhs_value = lhs_t[i];
    float rhs_value = rhs_t[i];
    lhs_t[i] = lhs_value * (1.0 - lrate) + rhs_value * lrate;
  }
});

array.def("track_vec2", [](soil::array& lhs, soil::array& rhs, const float lrate){

  auto lhs_t = lhs.as<soil::vec2>();
  auto rhs_t = rhs.as<soil::vec2>();

  for(size_t i = 0; i < lhs.shape().elem(); ++i){
    soil::vec2 lhs_value = lhs_t[i];
    soil::vec2 rhs_value = rhs_t[i];
    lhs_t[i] = lhs_value * (1.0f - lrate) + rhs_value * lrate;
  }
});

}

#endif