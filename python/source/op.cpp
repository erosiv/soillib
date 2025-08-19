#ifndef SOILLIB_PYTHON_LAYER
#define SOILLIB_PYTHON_LAYER

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/ndarray.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/function.h>

#include <soillib/core/types.hpp>

#include <soillib/core/operation.hpp>
#include <soillib/op/common.hpp>
#include <soillib/op/noise.hpp>
#include <soillib/op/normal.hpp>

#include <iostream>

#include "glm.hpp"

void bind_op(nb::module_& module) {

//
// Generic Buffer Reductions
//

module.def("min", [](const soil::tensor& tensor){
  return soil::select(tensor.type(), [&tensor]<std::floating_point S>() -> nb::object {
    return nb::cast(soil::min(tensor.as<S>()));
  });
});

module.def("max", [](const soil::tensor& tensor){
  return soil::select(tensor.type(), [&tensor]<std::floating_point S>() -> nb::object {
    return nb::cast(soil::max(tensor.as<S>()));
  });
});

module.def("clamp", [](soil::tensor& tensor, const float min, const float max){
  soil::select(tensor.type(), [&tensor, min, max]<std::same_as<float> S>() -> void {
    soil::clamp(tensor.as<S>(), min, max);
  });
});

//
// Generic Buffer Functions
//

module.def("cast", [](const soil::tensor& tensor, const soil::dtype type){
  if(tensor.type() == type){
    return nb::cast(tensor);
  }
  return soil::select(type, [&tensor]<std::floating_point To>() -> nb::object {
    return soil::select(tensor.type(), [&tensor]<std::floating_point From>() -> nb::object {
      soil::tensor tensor = soil::cast<To, From>(tensor.as<From>());
      return nb::cast(tensor);
    });
  });
});

module.def("copy", [](soil::tensor& lhs, const soil::tensor& rhs, soil::vec2 gmin, soil::vec2 gmax, soil::vec2 gscale, soil::vec2 wmin, soil::vec2 wmax, soil::vec2 wscale, float pscale){

  // Note: This supports copy between different buffer types.
  // The interior template selection just requires that the source
  // buffer's type can be converted to the target buffer's type.

  soil::select(lhs.type(), [&]<typename To>(){
    soil::select(rhs.type(), [&]<std::convertible_to<To> From>(){
      soil::copy<To, From>(lhs.as<To>(), rhs.as<From>(), gmin, gmax, gscale, wmin, wmax, wscale, pscale);
    });
  });
});

module.def("resize", [](const soil::tensor& rhs, const soil::shape shape){
  return soil::select(rhs.type(), [&rhs, shape]<typename S>() -> soil::tensor {
    return soil::tensor(soil::resize<S>(rhs.as<S>(), shape));
  });
});

module.def("resample", [](soil::tensor& target, const soil::tensor& source, const soil::vec3 t_scale, const soil::vec3 s_scale, const soil::vec2 pdiff){
  soil::select(target.type(), [&]<typename S>() {
    soil::resample<S>(target.as<S>(), source.as<S>(), t_scale, s_scale, pdiff);
  });
});

module.def("set", [](soil::tensor& lhs, const soil::tensor& rhs){

  if(lhs.type() != rhs.type())
    throw soil::error::mismatch_type(lhs.type(), rhs.type());
  
  if (lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());
  
  soil::select(lhs.type(), [&lhs, &rhs]<typename S>(){
    soil::set<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("set", [](soil::tensor& tensor, const nb::object value){
  soil::select(tensor.type(), [&tensor, &value]<typename S>(){
    auto tensor_t = tensor.as<S>();
    auto value_t = nb::cast<S>(value);
    soil::set<S>(tensor_t, value_t);
  });
});

module.def("add", [](soil::tensor& lhs, const soil::tensor& rhs){

  if(lhs.type() != rhs.type())
    throw soil::error::mismatch_type(lhs.type(), rhs.type());

  if (lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

  soil::select(lhs.type(), [&lhs, &rhs]<typename S>(){
    soil::add<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("add", [](soil::tensor& buffer, const nb::object value){
  soil::select(buffer.type(), [&buffer, &value]<typename S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    soil::add<S>(buffer_t, value_t);
  });
});

module.def("multiply", [](soil::tensor& lhs, const soil::tensor& rhs){
  
  if(lhs.type() != rhs.type())
    throw soil::error::mismatch_type(lhs.type(), rhs.type());

  if (lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

  soil::select(lhs.type(), [&lhs, &rhs]<typename S>(){
    soil::multiply<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("multiply", [](soil::tensor& buffer, const nb::object value){
  soil::select(buffer.type(), [&buffer, &value]<typename S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    soil::multiply<S>(buffer_t, value_t);
  });
});

//
// Noise Sampler Type
//

auto noise_t = nb::class_<soil::noise_param_t>(module, "noise_t");
noise_t.def(nb::init<>());
noise_t.def_rw("frequency", &soil::noise_param_t::frequency);
noise_t.def_rw("octaves", &soil::noise_param_t::octaves);
noise_t.def_rw("gain", &soil::noise_param_t::gain);
noise_t.def_rw("lacunarity", &soil::noise_param_t::lacunarity);
noise_t.def_rw("seed", &soil::noise_param_t::seed);
noise_t.def_rw("ext", &soil::noise_param_t::ext);

module.def("noise", [](const soil::shape shape, const soil::noise_param_t param){
  // note: seed is considered state. how can this be reflected here?
  return soil::noise(shape, param);
});

//
// Normal Map ?
//

module.def("normal", [](const soil::tensor& tensor, const soil::vec3 scale){

  if (tensor.host() != soil::CPU)
    throw soil::error::mismatch_host(soil::CPU, tensor.host());

  return soil::select(tensor.type(), [&]<std::floating_point T>(){
    return soil::op::normal(tensor.as<T>(), scale);
  });

});

}

#endif