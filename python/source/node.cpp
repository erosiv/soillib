#ifndef SOILLIB_PYTHON_LAYER
#define SOILLIB_PYTHON_LAYER

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/ndarray.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/function.h>

#include <soillib/core/types.hpp>

#include <soillib/core/node.hpp>

#include <soillib/node/cached.hpp>
#include <soillib/node/constant.hpp>
#include <soillib/node/computed.hpp>

#include <soillib/node/algorithm/noise.hpp>
#include <soillib/node/algorithm/normal.hpp>

#include "glm.hpp"

//! General Layer Binding Function
void bind_layer(nb::module_& module){

  //
  // Layer Wrapper Type
  //

  auto layer = nb::class_<soil::layer>(module, "layer");
  layer.def("type", &soil::layer::type);

  layer.def(nb::init<const soil::buffer>());
  layer.def(nb::init<soil::cached&&>());
  layer.def(nb::init<soil::constant&&>());
  layer.def(nb::init<soil::computed&&>());

  layer.def("buffer", [](soil::layer& layer){
    auto cached = std::get<soil::cached>(layer._layer);
    return soil::typeselect(cached.type(), [&cached]<typename T>() -> soil::buffer {
      return soil::buffer(cached.as<T>().buffer);
    });
  });

  layer.def("__call__", [](soil::layer& layer, const size_t index){
    return std::visit([index](auto&& args){
      return soil::typeselect(args.type(), [&args, index]<typename T>() -> nb::object {
        T value = args.template as<T>()(index);
        return nb::cast<T>(std::move(value));
      });
    }, layer._layer);
  });

  layer.def("numpy", [](soil::layer& layer, soil::index& index){
    
    return soil::indexselect(index.type(), [&]<typename I>() -> nb::object {

      auto index_t = index.as<I>();                 // Cast Index to Strict-Type
      soil::flat_t<I::n_dims> flat(index_t.ext());  // Hypothetical Flat Buffer

      //! \todo Remove this requirement, not actually necessary.
      auto cached = std::get<soil::cached>(layer._layer);

      return soil::typeselect(cached.type(), [&]<typename T>() -> nb::object {

        if constexpr(nb::detail::is_ndarray_scalar_v<T>){

          soil::cached_t<T> source = cached.as<T>();  // Source Buffer w. Index

          // Typed Buffer of Flat Size
          //! \todo make sure this is de-allocated correctly,
          //! i.e. the numpy buffer should perform a copy.
          soil::buffer_t<T>* buffer  = new soil::buffer_t<T>(flat.elem()); 

          // Fill w. NaN Value
          T value = std::numeric_limits<T>::quiet_NaN();
          for(size_t i = 0; i < buffer->elem(); ++i)
            buffer->operator[](i) = value;

          // Iterate over Flat Index
          for(const auto& pos: index_t.iter()){
            const size_t i = index_t.flatten(pos);
            buffer->operator[](flat.flatten(pos - index_t.min())) = source(i);
          }

          size_t shape[I::n_dims]{0};
          for(size_t d = 0; d < I::n_dims; ++d)
            shape[d] = flat[d];

          nb::ndarray<nb::numpy, T, nb::ndim<I::n_dims>> array(
            buffer->data(),
            I::n_dims,
            shape,
            nb::handle()
          );
          return nb::cast(std::move(array));

        } 
        
        //! \todo Make this Generic
        else if constexpr(std::same_as<T, soil::vec3>) {
          
          soil::cached_t<T> source = cached.as<T>();  // Source Buffer w. Index

          // Typed Buffer of Flat Size
          //! \todo make sure this is de-allocated correctly,
          //! i.e. the numpy buffer should perform a copy.
          soil::buffer_t<T>* buffer  = new soil::buffer_t<T>(flat.elem()); 

          // Fill w. NaN Value
          //! \todo automate the related NaN value determination
          //buffer->fill(T{std::numeric_limits<float>::quiet_NaN()});

          T value = T{std::numeric_limits<float>::quiet_NaN()};
          for(size_t i = 0; i < buffer->elem(); ++i)
            buffer->operator[](i) = value;

          // Iterate over Flat Index
          for(const auto& pos: index_t.iter()){
            const size_t i = index_t.flatten(pos);
            buffer->operator[](flat.flatten(pos - index_t.min())) = source(i);
          }

          size_t shape[I::n_dims + 1]{0};
          for(size_t d = 0; d < I::n_dims; ++d)
            shape[d] = flat[d];
          shape[I::n_dims] = 3;

          nb::ndarray<nb::numpy, float, nb::ndim<I::n_dims+1>> array(
            buffer->data(),
            I::n_dims+1,
            shape,
            nb::handle()
          );
          return nb::cast(std::move(array));

        } else {

          throw std::invalid_argument("can't convert type to numpy array");

        }
      
      });
    });
  });

  //
  // Cache-Valued Layer, i.e. Lookup Table
  //

  module.def("cached", [](const soil::buffer& buffer){
    return soil::layer(std::move(soil::cached(buffer)));
  });

  //
  // Constant-Valued Layer
  //

  module.def("constant", [](const soil::dtype type, const nb::object object){
    return soil::typeselect(type, [type, &object]<typename T>(){
      const T value = nb::cast<T>(object);
      return soil::layer(std::move(soil::constant(type, value)));
    });
  });

  //
  // Generic Computed Layer
  //

  module.def("computed", [](const soil::dtype type, const nb::callable object){
    return soil::typeselect(type, [type, &object]<typename T>(){
      using func_t = std::function<T(const size_t)>;
      func_t func = nb::cast<func_t>(object);
      return soil::layer(std::move(soil::computed(type, func)));
    });
  });

  //
  //
  //

  auto normal = nb::class_<soil::normal>(module, "normal");
  normal.def(nb::init<const soil::index&, const soil::layer&>());
  normal.def("full", &soil::normal::full);

  //
  // Noise Sampler Type
  //

  auto noise = nb::class_<soil::noise>(module, "noise");
  noise.def(nb::init<const soil::index, const float>());
  noise.def("full", &soil::noise::full);

  //
  // Special Layer-Based Operations
  //  These will be unified and expanded later!
  //

  layer.def("track_float", [](soil::layer& lhs, soil::layer& rhs, const float lrate){

    auto lhs_t = std::get<soil::cached>(lhs._layer).as<float>().buffer;
    auto rhs_t = std::get<soil::cached>(rhs._layer).as<float>().buffer;

    for(size_t i = 0; i < lhs_t.elem(); ++i){
      float lhs_value = lhs_t[i];
      float rhs_value = rhs_t[i];
      lhs_t[i] = lhs_value * (1.0 - lrate) + rhs_value * lrate;
    }

  });

  layer.def("track_vec2", [](soil::layer& lhs, soil::layer& rhs, const float lrate){

    auto lhs_t = std::get<soil::cached>(lhs._layer).as<soil::vec2>().buffer;
    auto rhs_t = std::get<soil::cached>(rhs._layer).as<soil::vec2>().buffer;

    for(size_t i = 0; i < lhs_t.elem(); ++i){
      soil::vec2 lhs_value = lhs_t[i];
      soil::vec2 rhs_value = rhs_t[i];
      lhs_t[i] = lhs_value * (1.0f - lrate) + rhs_value * lrate;
    }

  });

}

#endif