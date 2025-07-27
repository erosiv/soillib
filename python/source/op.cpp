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
// #include <soillib/op/flow.hpp>
#include <soillib/model/erosion.hpp>

#include <iostream>

#include "glm.hpp"

void bind_op(nb::module_& module){

//
// Generic Buffer Reductions
//

module.def("cast", [](const soil::buffer& buf, const soil::dtype type){
  if(buf.type() == type){
    return nb::cast(buf);
  }
  return soil::select(type, [&buf]<std::floating_point To>() -> nb::object {
    return soil::select(buf.type(), [&buf]<std::floating_point From>() -> nb::object {
      soil::buffer buffer = soil::cast<To, From>(buf.as<From>());
      return nb::cast(buffer);
    });
  });
});

module.def("min", [](const soil::buffer& buf){
  return soil::select(buf.type(), [&buf]<std::floating_point S>() -> nb::object {
    return nb::cast(soil::op::min(buf.as<S>()));
  });
});

module.def("max", [](const soil::buffer& buf){
  return soil::select(buf.type(), [&buf]<std::floating_point S>() -> nb::object {
    return nb::cast(soil::op::max(buf.as<S>()));
  });
});

module.def("clamp", [](soil::buffer& buf, const float min, const float max){
  soil::select(buf.type(), [&]<std::same_as<float> S>() -> void {
    soil::op::clamp(buf.as<S>(), min, max);
  });
});

//
// Generic Buffer Functions
//

module.def("copy", [](soil::buffer& lhs, const soil::buffer& rhs, soil::vec2 gmin, soil::vec2 gmax, soil::vec2 gscale, soil::vec2 wmin, soil::vec2 wmax, soil::vec2 wscale, float pscale){

  // Note: This supports copy between different buffer types.
  // The interior template selection just requires that the source
  // buffer's type can be converted to the target buffer's type.

  soil::select(lhs.type(), [&]<typename To>(){
    soil::select(rhs.type(), [&]<std::convertible_to<To> From>(){
      soil::op::copy<To, From>(lhs.as<To>(), rhs.as<From>(), gmin, gmax, gscale, wmin, wmax, wscale, pscale);
    });
  });
});

module.def("resize", [](soil::buffer& lhs, const soil::buffer& rhs, soil::ivec2 out, soil::ivec2 in){
  if(lhs.type() != rhs.type())
    throw soil::error::mismatch_type(lhs.type(), rhs.type());
  soil::select(lhs.type(), [&lhs, &rhs, in, out]<typename S>(){
    soil::resize<S>(lhs.as<S>(), rhs.as<S>(), out, in);
  });
});









module.def("set", [](soil::buffer& lhs, const soil::buffer& rhs){

  if(lhs.type() != rhs.type())
    throw soil::error::mismatch_type(lhs.type(), rhs.type());
  
  if (lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());
  
  soil::select(lhs.type(), [&lhs, &rhs]<typename S>(){
    soil::op::set<S>(lhs.as<S>(), rhs.as<S>());
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

  if (lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

  soil::select(lhs.type(), [&lhs, &rhs]<typename S>(){
    soil::op::add<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("add", [](soil::buffer& buffer, const nb::object value){
  soil::select(buffer.type(), [&buffer, &value]<typename S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    soil::op::add<S>(buffer_t, value_t);
  });
});

module.def("multiply", [](soil::buffer& lhs, const soil::buffer& rhs){
  
  if(lhs.type() != rhs.type())
    throw soil::error::mismatch_type(lhs.type(), rhs.type());

  if (lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

  soil::select(lhs.type(), [&lhs, &rhs]<typename S>(){
    soil::op::multiply<S>(lhs.as<S>(), rhs.as<S>());
  });
});

module.def("multiply", [](soil::buffer& buffer, const nb::object value){
  soil::select(buffer.type(), [&buffer, &value]<typename S>(){
    auto buffer_t = buffer.as<S>();
    auto value_t = nb::cast<S>(value);
    soil::op::multiply<S>(buffer_t, value_t);
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
  return soil::noise::make_buffer(shape, param);
});

//
// Normal Map ?
//

module.def("normal", [](const soil::buffer& buffer, const soil::shape& shape, const soil::vec3 scale){
  return soil::normal::operator()(buffer, shape, scale);
});

//
// Erosion Kernels
//

auto param_t = nb::class_<soil::param_t>(module, "param_t");
param_t.def(nb::init<>());
param_t.def_rw("samples", &soil::param_t::samples);
param_t.def_rw("maxage", &soil::param_t::maxage);
param_t.def_rw("timeStep", &soil::param_t::timeStep);

param_t.def_rw("critSlope", &soil::param_t::critSlope);
param_t.def_rw("debrisCreepRate", &soil::param_t::debrisCreepRate);
param_t.def_rw("debrisDepositionRate", &soil::param_t::debrisDepositionRate);
param_t.def_rw("debrisSuspensionRate", &soil::param_t::debrisSuspensionRate);
param_t.def_rw("debrisYieldStress", &soil::param_t::debrisYieldStress);
param_t.def_rw("debrisDensity", &soil::param_t::debrisDensity);
param_t.def_rw("debrisViscosity", &soil::param_t::debrisViscosity);
param_t.def_rw("debrisBedShear", &soil::param_t::debrisBedShear);

param_t.def_rw("uplift", &soil::param_t::uplift);
param_t.def_rw("rainfall", &soil::param_t::rainfall);
param_t.def_rw("evapRate", &soil::param_t::evapRate);
param_t.def_rw("depositionRate", &soil::param_t::depositionRate);
param_t.def_rw("suspensionRate", &soil::param_t::suspensionRate);
param_t.def_rw("fluvialExponent", &soil::param_t::fluvialExponent);

param_t.def_rw("gravity", &soil::param_t::gravity);
param_t.def_rw("viscosity", &soil::param_t::viscosity);
param_t.def_rw("bedShear", &soil::param_t::bedShear);
param_t.def_rw("lrate", &soil::param_t::lrate);
param_t.def_rw("exitSlope", &soil::param_t::exitSlope);

param_t.def_rw("force", &soil::param_t::force);

//
// Map Data-Structure
//

auto map_t = nb::class_<soil::map_t>(module, "map_t");
map_t.def(nb::init<soil::shape, soil::vec3>());
map_t.def_ro("scale", &soil::map_t::scale);

map_t.def_prop_rw("height",
[](soil::map_t& map){
  return soil::buffer(map.height);
},[](soil::map_t& map, soil::buffer buffer){
  map.height = buffer.as<float>();
});

map_t.def_prop_rw("sediment",
[](soil::map_t& map){
  return soil::buffer(map.sediment);
},[](soil::map_t& map, soil::buffer buffer){
  map.sediment = buffer.as<float>();
});

map_t.def_prop_rw("uplift",
[](soil::map_t& map){
  return soil::buffer(map.uplift);
},[](soil::map_t& map, soil::buffer buffer){
  map.uplift = buffer.as<float>();
});

map_t.def_prop_rw("rainfall",
[](soil::map_t& map){
  return soil::buffer(map.rainfall);
},[](soil::map_t& map, soil::buffer buffer){
  map.rainfall = buffer.as<float>();
});

//
// Tracking Fields
//

auto data_t = nb::class_<soil::data_t>(module, "data_t");
data_t.def(nb::init<>());
data_t.def(nb::init<const size_t>());

data_t.def_prop_rw("discharge",
  [](soil::data_t& model){
    return soil::buffer(model.discharge);
},[](soil::data_t& model, soil::buffer buffer){
    model.discharge = buffer.as<float>();
});

data_t.def_prop_rw("momentum",
  [](soil::data_t& model){
    return soil::buffer(model.momentum);
},[](soil::data_t& model, soil::buffer buffer){
    model.momentum = buffer.as<soil::vec2>();
});

data_t.def_prop_rw("mass",
  [](soil::data_t& model){
    return soil::buffer(model.mass);
},[](soil::data_t& model, soil::buffer buffer){
    model.mass = buffer.as<float>();
});

data_t.def_prop_rw("debris",
  [](soil::data_t& model){
    return soil::buffer(model.debris);
},[](soil::data_t& model, soil::buffer buffer){
    model.debris = buffer.as<float>();
});

data_t.def_prop_rw("debris_momentum",
  [](soil::data_t& model){
    return soil::buffer(model.debris_momentum);
},[](soil::data_t& model, soil::buffer buffer){
    model.debris_momentum = buffer.as<soil::vec2>();
});

module.def("erode", soil::erode);

// note: consider how to implement this deferred using the nodes
// direct computation? immediate evaluation...

/*
module.def("flow", [](const soil::buffer& buffer, const soil::index& index){
  return soil::flow(buffer, index);
});

module.def("direction", [](const soil::buffer& buffer, const soil::index& index){
  return soil::direction(buffer, index);
});

module.def("accumulation", [](const soil::buffer& buffer, const soil::index& index, int iterations, int samples){
  return soil::accumulation(buffer, index, iterations, samples);
});

module.def("accumulation_weighted", [](const soil::buffer& buffer, const soil::buffer& weights, const soil::index& index, int iterations, int samples, bool reservoir){
  return soil::accumulation(buffer, weights, index, iterations, samples, reservoir);
});

module.def("accumulation_exhaustive", [](const soil::buffer& buffer, const soil::index& index){
  return soil::accumulation_exhaustive(buffer, index);
});

module.def("accumulation_exhaustive_weighted", [](const soil::buffer& buffer, const soil::index& index, const soil::buffer& weights){
  return soil::accumulation_exhaustive(buffer, index, weights);
});

module.def("upstream", [](const soil::buffer& buffer, const soil::index& index, const glm::ivec2 target){
  return soil::upstream(buffer, index, target);
});

module.def("distance", [](const soil::buffer& buffer, const soil::index& index, const glm::ivec2 target){
  return soil::distance(buffer, index, target);
});
*/

//
// Point-Based Operations
//

/*
module.def("sampleN", [](const soil::index& index, const size_t N){
  return soil::sample_N(index, N);
});

module.def("sample_halton", [](const soil::index& index, const size_t N){
  return soil::sample_halton(index, N);
});

module.def("sample_lerp", [](const soil::buffer& field, const soil::index& index, const soil::buffer& pos){
  return soil::sample_lerp(field, index, pos);
});

module.def("sample_grad", [](const soil::buffer& field, const soil::index& index, const soil::buffer& pos){
  return soil::sample_grad(field, index, pos);
});

module.def("concat", [](const soil::buffer& a, const soil::buffer& b){
  return soil::concat(a, b);
});

module.def("select_index", [](const soil::buffer& source, const soil::buffer& index){
  return soil::select_index(source, index);
});
*/

}

#endif