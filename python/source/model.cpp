#ifndef SOILLIB_PYTHON_MODEL
#define SOILLIB_PYTHON_MODEL

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <soillib/model/erosion.hpp>
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
  return soil::tensor(map.height);
},[](soil::map_t& map, soil::tensor tensor){
  map.height = tensor.as<float>();
});

map_t.def_prop_rw("sediment",
[](soil::map_t& map){
  return soil::tensor(map.sediment);
},[](soil::map_t& map, soil::tensor tensor){
  map.sediment = tensor.as<float>();
});

map_t.def_prop_rw("uplift",
[](soil::map_t& map){
  return soil::tensor(map.uplift);
},[](soil::map_t& map, soil::tensor tensor){
  map.uplift = tensor.as<float>();
});

map_t.def_prop_rw("rainfall",
[](soil::map_t& map){
  return soil::tensor(map.rainfall);
},[](soil::map_t& map, soil::tensor tensor){
  map.rainfall = tensor.as<float>();
});

//
// Tracking Fields
//

auto data_t = nb::class_<soil::data_t>(module, "data_t");
data_t.def(nb::init<>());
data_t.def(nb::init<const soil::shape>());

data_t.def_prop_rw("discharge",
  [](soil::data_t& model){
    return soil::tensor(model.discharge);
},[](soil::data_t& model, soil::tensor tensor){
    model.discharge = tensor.as<float>();
});

data_t.def_prop_rw("momentum",
  [](soil::data_t& model){
    return soil::tensor(model.momentum);
},[](soil::data_t& model, soil::tensor tensor){
    model.momentum = tensor.as<float>();
});

data_t.def_prop_rw("mass",
  [](soil::data_t& model){
    return soil::tensor(model.mass);
},[](soil::data_t& model, soil::tensor tensor){
    model.mass = tensor.as<float>();
});

data_t.def_prop_rw("debris",
  [](soil::data_t& model){
    return soil::tensor(model.debris);
},[](soil::data_t& model, soil::tensor tensor){
    model.debris = tensor.as<float>();
});

data_t.def_prop_rw("debris_momentum",
  [](soil::data_t& model){
    return soil::tensor(model.debris_momentum);
},[](soil::data_t& model, soil::tensor tensor){
    model.debris_momentum = tensor.as<float>();
});

module.def("erode", soil::erode);

// note: consider how to implement this deferred using the nodes
// direct computation? immediate evaluation...

/*
module.def("flow", [](const soil::tensor& tensor, const soil::index& index){
  return soil::flow(tensor, index);
});

module.def("direction", [](const soil::tensor& tensor, const soil::index& index){
  return soil::direction(tensor, index);
});

module.def("accumulation", [](const soil::tensor& tensor, const soil::index& index, int iterations, int samples){
  return soil::accumulation(tensor, index, iterations, samples);
});

module.def("accumulation_weighted", [](const soil::tensor& tensor, const soil::tensor& weights, const soil::index& index, int iterations, int samples, bool reservoir){
  return soil::accumulation(tensor, weights, index, iterations, samples, reservoir);
});

module.def("accumulation_exhaustive", [](const soil::tensor& tensor, const soil::index& index){
  return soil::accumulation_exhaustive(tensor, index);
});

module.def("accumulation_exhaustive_weighted", [](const soil::tensor& tensor, const soil::index& index, const soil::tensor& weights){
  return soil::accumulation_exhaustive(tensor, index, weights);
});

module.def("upstream", [](const soil::tensor& tensor, const soil::index& index, const glm::ivec2 target){
  return soil::upstream(tensor, index, target);
});

module.def("distance", [](const soil::tensor& tensor, const soil::index& index, const glm::ivec2 target){
  return soil::distance(tensor, index, target);
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

module.def("sample_lerp", [](const soil::tensor& field, const soil::index& index, const soil::tensor& pos){
  return soil::sample_lerp(field, index, pos);
});

module.def("sample_grad", [](const soil::tensor& field, const soil::index& index, const soil::tensor& pos){
  return soil::sample_grad(field, index, pos);
});

module.def("concat", [](const soil::tensor& a, const soil::tensor& b){
  return soil::concat(a, b);
});

module.def("select_index", [](const soil::tensor& source, const soil::tensor& index){
  return soil::select_index(source, index);
});
*/

}

#endif