#ifndef SOILLIB_PYTHON_MODEL
#define SOILLIB_PYTHON_MODEL

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <nanobind/ndarray.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/function.h>

#include <soillib/core/types.hpp>

#include <soillib/core/operation.hpp>
#include <soillib/op/common.hpp>
#include <soillib/op/normal.hpp>
// #include <soillib/op/flow.hpp>
#include <soillib/model/erosion.hpp>

#include <iostream>

#include "glm.hpp"

void bind_model(nb::module_& module){

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