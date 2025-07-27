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
#include <soillib/core/shape.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/util/error.hpp>

#include <soillib/op/common.hpp>

#include "glm.hpp"

template<typename T> struct make_numpy; //!< Buffer to Numpy Exporter
template<typename T> struct make_torch; //!< Buffer to PyTorch Exporter

//
//
//

//! General Util Binding Function
void bind_buffer(nb::module_& module){

//
// Buffer Type Binding
//

// Device Enumerator

nb::enum_<soil::host_t>(module, "host")
  .value("cpu", soil::host_t::CPU)
  .value("gpu", soil::host_t::GPU)
  .export_values();

// Buffer Type

auto buffer = nb::class_<soil::buffer>(module, "buffer");
buffer.def(nb::init<>());
buffer.def(nb::init<const soil::dtype, const size_t>());
buffer.def(nb::init<const soil::dtype, const size_t, const soil::host_t>());

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

buffer.def("__setitem__", [](soil::buffer& buffer, const nb::slice& slice, const nb::object value){

  const size_t elem = buffer.elem();
  Py_ssize_t start, stop, step;
  if(PySlice_GetIndices(slice.ptr(), elem, &start, &stop, &step) != 0)
    throw std::runtime_error("slice is invalid!");

  soil::select(buffer.type(), [&]<typename S>(){
    auto buffer_t = buffer.as<S>();           // Assignable Strict-Type Buffer
    const auto value_t = nb::cast<S>(value);  // Assignable Value
    soil::set(buffer_t, value_t, start, stop, step);
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

buffer.def("numpy", [](soil::buffer& buffer, soil::shape& shape){
  if(buffer.host() == soil::host_t::CPU){
    return soil::select(buffer.type(), [&buffer, &shape]<typename T>() -> nb::object {
      return make_numpy<T>::operator()(buffer, shape);
    });
  }
  throw soil::error::unsupported_host(soil::host_t::CPU, buffer.host());
});

buffer.def("torch", [](soil::buffer& buffer){
  if(buffer.host() == soil::host_t::GPU){
    return soil::select(buffer.type(), [&buffer]<typename T>() -> nb::object {
      return make_torch<T>::operator()(buffer);
    });
  }
  throw soil::error::unsupported_host(soil::host_t::GPU, buffer.host());
});

buffer.def("torch", [](soil::buffer& buffer, soil::shape& shape){
  if(buffer.host() == soil::host_t::GPU){
    return soil::select(buffer.type(), [&buffer, &shape]<typename T>() -> nb::object {
      return make_torch<T>::operator()(buffer, shape);
    });
  }
  throw soil::error::unsupported_host(soil::host_t::GPU, buffer.host());
});

//
// Construct Buffer from Numpy
//

//! \note this always performs a copy, it doesn't keep the object alive.
buffer.def_static("from_numpy", [](const nb::object& object){

  auto array = nb::cast<nb::ndarray<nb::numpy>>(object);

  if(array.dtype() == nb::dtype<float>()){

    const size_t size = array.size();
    const float* data = (float*)array.data();
    auto buffer_t = soil::buffer_t<float>(size, soil::host_t::CPU);

    for(size_t i = 0; i < size; ++i)
      buffer_t[i] = data[i];

    return std::move(soil::buffer(std::move(buffer_t)));

  }
  
  else if(array.dtype() == nb::dtype<double>()){

    const size_t size = array.size();
    const double* data = (double*)array.data();
    auto buffer_t = soil::buffer_t<double>(size, soil::host_t::CPU);

    for(size_t i = 0; i < size; ++i)
      buffer_t[i] = data[i];

    return std::move(soil::buffer(std::move(buffer_t)));

  }
  else {
    throw std::runtime_error("type not supported");
  }

});

//
// Construct Buffer from Pytorch
//

buffer.def_static("from_torch", [](nb::object& object){

  auto array = nb::cast<nb::ndarray<nb::pytorch>>(object);

  if(array.dtype() == nb::dtype<float>()){

    const size_t size = array.size();
    float* data = (float*)array.data();
    auto buffer_t = soil::buffer_t<float>(size, soil::host_t::GPU);
    const auto view_t = soil::buffer_t<float>(data, size, soil::host_t::GPU);

    soil::op::set(buffer_t, view_t);
    return soil::buffer(buffer_t);

  }
  
  else if(array.dtype() == nb::dtype<double>()){

    const size_t size = array.size();
    double* data = (double*)array.data();
    auto buffer_t = soil::buffer_t<double>(size, soil::host_t::CPU);
    const auto view_t = soil::buffer_t<double>(data, size, soil::host_t::GPU);

    soil::op::set(buffer_t, view_t);
    return soil::buffer(buffer_t);

  }
  else {
    throw std::runtime_error("type not supported");
  }

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
      using V = soil::typedesc<T>::value_t;

      size_t shape[2] = { source.elem(), D };
      nb::ndarray<nb::numpy, V, nb::ndim<D>> array(
        source.data(),
        2,
        shape,
        nb::find(buffer)
      );
      return nb::cast(std::move(array));

    }

  }

  static nb::object operator()(soil::buffer& buffer, soil::shape& shape){

    if constexpr(nb::detail::is_ndarray_scalar_v<T>){

    // Target Buffer (Heap)
    soil::buffer_t<T> source = buffer.as<T>();
    soil::buffer_t<T>* target  = new soil::buffer_t<T>(source.elem(), source.host()); 
    nb::capsule owner(target, [](void *p) noexcept {
      delete (soil::buffer_t<T>*)p;
    });
    soil::op::set(*target, source);

    size_t _shape[4]{0};
    for(size_t d = 0; d < 4; ++d)
      _shape[d] = shape[d];

    if(shape.dim == 1){
      nb::ndarray<nb::numpy, T, nb::ndim<1>> array(
        target->data(),
        1,
        _shape,
        owner
      );
      return nb::cast(std::move(array));
    }
    else if(shape.dim == 2){
      nb::ndarray<nb::numpy, T, nb::ndim<2>> array(
        target->data(),
        2,
        _shape,
        owner
      );
      return nb::cast(std::move(array));
    }
    else if(shape.dim == 3){
      nb::ndarray<nb::numpy, T, nb::ndim<3>> array(
        target->data(),
        3,
        _shape,
        owner
      );
      return nb::cast(std::move(array));
    }
    else {
      nb::ndarray<nb::numpy, T, nb::ndim<4>> array(
        target->data(),
        4,
        _shape,
        owner
      );
      return nb::cast(std::move(array));
    }

    } else {
      throw std::invalid_argument("buffer type cannot be converted");
    }

  }

};

//
// Torch Buffer Generator
//

template<typename T>
struct make_torch {

  static nb::object operator()(soil::buffer& buffer){

  // buffer arrives as a python object. we tie the lifetime of the buffer to
  // the numpy array, so that the memory is always accessible / not deleted.

    soil::buffer_t<T> source = buffer.as<T>();

    if constexpr(nb::detail::is_ndarray_scalar_v<T>){

      size_t shape[1] = { source.elem() };
      nb::ndarray<nb::pytorch, nb::device::cuda, T, nb::ndim<1>> array(
        source.data(),    // raw data pointer
        1,                // number of dimensions
        shape,            // shape of array
        nb::find(buffer), // lifetime guarantee
        nullptr,
        nb::dtype<T>(),
        nb::device::cuda::value
      );
      return nb::cast(std::move(array));

    } else {
      //! \todo add a concept to explicitly test for vector types
      
      constexpr int D = T::length();
      using V = soil::typedesc<T>::value_t;
      
      size_t shape[2] = { source.elem(), D };
      nb::ndarray<nb::pytorch, nb::device::cuda, V, nb::ndim<D>> array(
        source.data(),
        2,
        shape,
        nb::find(buffer),
        nullptr,
        nb::dtype<V>(),
        nb::device::cuda::value
      );
      return nb::cast(std::move(array));

    } // else throw std::invalid_argument("can't convert non-scalar buffer type (yet)");

  }

  // note: this is generally bad, because it basically says that re-sampling
  //  can only happen with a CPU tensor type?
  //  so we really have to implement a re-sample kernel first.
  // ... so let's do that I guess???

  static nb::object operator()(soil::buffer& buffer, soil::shape& shape){

    if constexpr(nb::detail::is_ndarray_scalar_v<T>){

    soil::buffer_t<T> source = buffer.as<T>();
    soil::buffer_t<T>* target = new soil::buffer_t<T>(source.elem(), source.host());
    nb::capsule owner(target, [](void *p) noexcept {
      delete (soil::buffer_t<T>*)p;
    });
    soil::op::set(*target, source);

    size_t _shape[4]{0};
    for(size_t d = 0; d < 4; ++d)
      _shape[d] = shape[d];

    if(shape.dim == 1){
      nb::ndarray<nb::pytorch, T, nb::ndim<1>> array(
        target->data(),
        1,
        _shape,
        owner,
        nullptr,
        nb::dtype<T>(),
        nb::device::cuda::value
      );
      return nb::cast(std::move(array));
    }
    else if(shape.dim == 2){
      nb::ndarray<nb::pytorch, T, nb::ndim<2>> array(
        target->data(),
        2,
        _shape,
        owner,
        nullptr,
        nb::dtype<T>(),
        nb::device::cuda::value
      );
      return nb::cast(std::move(array));
    }
    else if(shape.dim == 3){
      nb::ndarray<nb::pytorch, T, nb::ndim<3>> array(
        target->data(),
        3,
        _shape,
        owner,
        nullptr,
        nb::dtype<T>(),
        nb::device::cuda::value
      );
      return nb::cast(std::move(array));
    }
    else {
      nb::ndarray<nb::pytorch, T, nb::ndim<4>> array(
        target->data(),
        4,
        _shape,
        owner,
        nullptr,
        nb::dtype<T>(),
        nb::device::cuda::value
      );
      return nb::cast(std::move(array));
    }

    }
    else throw std::invalid_argument("buffer type cannot be converted");

  }

};

#endif