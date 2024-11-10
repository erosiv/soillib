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

#include <soillib/node/common.hpp>

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

template<typename T>
struct make_numpy;

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

nb::enum_<soil::host_t>(module, "host")
  .value("cpu", soil::host_t::CPU)
  .value("gpu", soil::host_t::GPU)
  .export_values();

auto buffer = nb::class_<soil::buffer>(module, "buffer");
buffer.def(nb::init<>());
buffer.def(nb::init<const soil::dtype, const size_t>());

buffer.def_prop_ro("type", &soil::buffer::type);
buffer.def_prop_ro("elem", &soil::buffer::elem);
buffer.def_prop_ro("size", &soil::buffer::size);
buffer.def_prop_ro("host", &soil::buffer::host);

// Device-Switching Functions:
// Return a Copy of the Buffer Directly
// Note: Buffer handles internal reference counting.
// Therefore, the copy here is fine. Swap is in-place.

buffer.def("cpu", [](soil::buffer& buffer){
  soil::select(buffer.type(), [&buffer]<typename T>(){
    buffer.as<T>().to_cpu();
  });
  return buffer;
});

buffer.def("gpu", [](soil::buffer& buffer){
  soil::select(buffer.type(), [&buffer]<typename T>(){
    buffer.as<T>().to_gpu();
  });
  return buffer;
});

// 

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

buffer.def("__setitem__", [](soil::buffer& buffer, const nb::slice& slice, const nb::object value){

  soil::select(buffer.type(), [&buffer, &slice, &value]<typename S>(){

    auto buffer_t = buffer.as<S>();           // Assignable Strict-Type Buffer
    const auto value_t = nb::cast<S>(value);  // Assignable Value

    // Read Slice:
    Py_ssize_t start, stop, step;
    if(PySlice_GetIndices(slice.ptr(), buffer_t.elem(), &start, &stop, &step) != 0)
      throw std::runtime_error("slice is invalid!");
    
    // Assign Values!
    for(int index = start; index < stop; index += step)
      buffer_t[index] = value_t;
  
  });
});

//
// Generic Buffer Functions
//

module.def("set", [](soil::buffer& lhs, const soil::buffer& rhs){
  if(lhs.type() != rhs.type())
    throw soil::error::mismatch_type(lhs.type(), rhs.type());
  soil::select(lhs.type(), [&lhs, &rhs]<typename S>(){
    soil::set<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("set", [](soil::buffer& buffer, const nb::object value){
  soil::select(buffer.type(), [&buffer, &value]<typename S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    soil::set<S>(buffer_t, value_t);
  });
});

module.def("add", [](soil::buffer& lhs, const soil::buffer& rhs){
  if(lhs.type() != rhs.type())
    throw soil::error::mismatch_type(lhs.type(), rhs.type());
  soil::select(lhs.type(), [&lhs, &rhs]<typename S>(){
    soil::add<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("add", [](soil::buffer& buffer, const nb::object value){
  soil::select(buffer.type(), [&buffer, &value]<typename S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    soil::add<S>(buffer_t, value_t);
  });
});

module.def("multiply", [](soil::buffer& lhs, const soil::buffer& rhs){
  if(lhs.type() != rhs.type())
    throw soil::error::mismatch_type(lhs.type(), rhs.type());
  soil::select(lhs.type(), [&lhs, &rhs]<typename S>(){
    soil::multiply<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("multiply", [](soil::buffer& buffer, const nb::object value){
  soil::select(buffer.type(), [&buffer, &value]<typename S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    soil::multiply<S>(buffer_t, value_t);
  });
});

//
// External Library Interop Interface
//

// Note: Memory is shared, not copied.
// The lifetimes of the objects are managed
// so that the memory is not deleted.

buffer.def("numpy", [](soil::buffer& buffer){
  if(buffer.host() == soil::host_t::CPU){
    return soil::select(buffer.type(), [&buffer]<typename T>() -> nb::object {
      return make_numpy<T>::operator()(buffer);
    });
  }
  throw soil::error::unsupported_host(soil::host_t::CPU, buffer.host());
});

buffer.def("numpy", [](soil::buffer& buffer, soil::index& index){
  if(buffer.host() == soil::host_t::CPU){
    return soil::select(buffer.type(), [&buffer, &index]<typename T>() -> nb::object {
      return make_numpy<T>::operator()(buffer, index);
    });
  }
  throw soil::error::unsupported_host(soil::host_t::CPU, buffer.host());
});

//
// Construct Buffer from Numpy
//

buffer.def("from_numpy", [](soil::buffer& buffer, const nb::object& object){

  /*
  
  // I suppose that this does not necessarily matter.
  // we could in principle have either torch or numpy.

  // We also need a way to select the underlying type of the
  // numpy buffer... of course.

  // and tie the lifetimes!

  if(buffer.host() == soil::host_t::CPU){
    soil::select(buffer.type(), [&buffer, &object]<typename T>() {
      const auto array = nb::cast<nb::ndarray<nb::numpy, T>>(object);
      // const soil::buffer_t<T> source {
      //   array.
      // };
      // note: tie the object lifetimes somehow!
      buffer = std::move(soil::buffer(std::move(source)));
    });
  }
  throw soil::error::unsupported_host(soil::host_t::CPU, buffer.host());
  */

});

}

//
// Numpy Buffer from Type Buffer Generator
//

template<typename T>
struct make_numpy {

  static nb::object operator()(soil::buffer& buffer){

  // buffer arrives as a python object. we tie the lifetime of the buffer to
  // the numpy array, so that the memory is always accessible / not deleted.

    soil::buffer_t<T> source = buffer.as<T>();

    if constexpr(nb::detail::is_ndarray_scalar_v<T>){

      size_t shape[1] = { source.elem() };
      nb::ndarray<nb::numpy, T, nb::ndim<1>> array(
        source.data(),    // raw data pointer
        1,                // number of dimensions
        shape,            // shape of array
        nb::find(buffer)  // lifetime guarantee
      );
      return nb::cast(std::move(array));

    
    } else {
      //! \todo add a concept to explicitly test for vector types
      
      constexpr int D = T::length();
      size_t shape[2] = { source.elem(), D };
      nb::ndarray<nb::numpy, typename T::value_type, nb::ndim<D>> array(
        source.data(),
        2,
        shape,
        nb::find(buffer)
      );
      return nb::cast(std::move(array));

    } // else throw std::invalid_argument("can't convert non-scalar buffer type (yet)");

  }

  static nb::object operator()(soil::buffer& buffer, soil::index& index){

    soil::buffer_t<T> source = buffer.as<T>();

    return soil::select(index.type(), [&source, &index]<typename I>() -> nb::object {

      auto index_t = index.as<I>();                 // Cast Index to Strict-Type
      soil::flat_t<I::n_dims> flat(index_t.ext());  // Hypothetical Flat Buffer

      soil::buffer_t<T>* target  = new soil::buffer_t<T>(flat.elem()); // Target Buffer (Heap)
      nb::capsule owner(target, [](void *p) noexcept {
        delete (soil::buffer_t<T>*)p;
      });

      if constexpr(nb::detail::is_ndarray_scalar_v<T>){

        T value = T{std::numeric_limits<T>::quiet_NaN()};
        soil::set<T>(*target, value);

        for(const auto& pos: index_t.iter()){
          const size_t i = index_t.flatten(pos);
          target->operator[](flat.flatten(pos - index_t.min())) = source[i];
        }

        size_t shape[I::n_dims]{0};
        for(size_t d = 0; d < I::n_dims; ++d)
          shape[d] = flat[d];

        nb::ndarray<nb::numpy, T, nb::ndim<I::n_dims>> array(
          target->data(),
          I::n_dims,
          shape,
          owner
        );
        return nb::cast(std::move(array));

      } else { //! \todo add a concept to explicitly test for vector types

        constexpr int D = T::length();
        using V = typename T::value_type;

        T value = T{std::numeric_limits<V>::quiet_NaN()};
        soil::set<T>(*target, value);

        for(const auto& pos: index_t.iter()){
          const size_t i = index_t.flatten(pos);
          target->operator[](flat.flatten(pos - index_t.min())) = source[i];
        }

        size_t shape[I::n_dims + 1]{0};
        for(size_t d = 0; d < I::n_dims; ++d)
          shape[d] = flat[d];
        shape[I::n_dims] = D;

        nb::ndarray<nb::numpy, float, nb::ndim<I::n_dims+1>> array(
          target->data(),
          I::n_dims+1,
          shape,
          owner
        );
        return nb::cast(std::move(array));

      } // else throw std::invalid_argument("can't convert non-scalar buffer type (yet)");

    });

  }

};

#endif