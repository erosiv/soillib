#ifndef SOILLIB_PYTHON_MODEL
#define SOILLIB_PYTHON_MODEL

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <soillib/model/path/erosion.hpp>
#include <soillib/model/graph/graph.hpp>
#include <soillib/model/filter/filter.hpp>
#include <soillib/model/path/path.hpp>
#include <soillib/model/grad/grad.hpp>
#include <soillib/model/sculpt/sculpt.hpp>
#include <soillib/op/noise.hpp>
#include "glm.hpp"

using namespace nb::literals;

void bind_model(nb::module_& module){

//
// Erosion Kernels
//

auto param_t = nb::class_<soil::param_t>(module, "param_t");
param_t.def(nb::init<>());
param_t.def_rw("maxage", &soil::param_t::maxage);
param_t.def_rw("timeStep", &soil::param_t::timeStep);
param_t.def_rw("lrate", &soil::param_t::lrate);
param_t.def_rw("exitSlope", &soil::param_t::exitSlope);
param_t.def_rw("critSlope", &soil::param_t::critSlope);

param_t.def_rw("uplift", &soil::param_t::uplift);
param_t.def_rw("rainfall", &soil::param_t::rainfall);

param_t.def_rw("evapRate", &soil::param_t::evapRate);
param_t.def_rw("depositionRate", &soil::param_t::depositionRate);
param_t.def_rw("suspensionRate", &soil::param_t::suspensionRate);
param_t.def_rw("fluvialExponent", &soil::param_t::fluvialExponent);

param_t.def_rw("debrisDepositionRate", &soil::param_t::debrisDepositionRate);
param_t.def_rw("debrisSuspensionRate", &soil::param_t::debrisSuspensionRate);
param_t.def_rw("debrisYieldStress", &soil::param_t::debrisYieldStress);
param_t.def_rw("debrisViscosity", &soil::param_t::debrisViscousStress);

param_t.def_rw("force", &soil::param_t::force);

// Momentum Parameterization

auto momentum_param_t = nb::class_<soil::momentum_param_t>(module, "momentum_param_t");
momentum_param_t.def(nb::init<>());
momentum_param_t.def_rw("gravity", &soil::momentum_param_t::gravity);
momentum_param_t.def_rw("density", &soil::momentum_param_t::density);
momentum_param_t.def_rw("viscosity", &soil::momentum_param_t::viscosity);
momentum_param_t.def_rw("bedShear", &soil::momentum_param_t::bedShear);

/*
//
// Map Data-Structure
//

auto map_t = nb::class_<soil::map_t>(module, "map_t");
map_t.def(nb::init<silt::shape, silt::vec3>());
map_t.def_ro("scale", &soil::map_t::scale);

map_t.def_prop_rw("height",
[](soil::map_t& map){
  return silt::tensor(map.height);
},[](soil::map_t& map, silt::tensor tensor){
  map.height = tensor.as<float>();
});

map_t.def_prop_rw("sediment",
[](soil::map_t& map){
  return silt::tensor(map.sediment);
},[](soil::map_t& map, silt::tensor tensor){
  map.sediment = tensor.as<float>();
});

map_t.def_prop_rw("uplift",
[](soil::map_t& map){
  return silt::tensor(map.uplift);
},[](soil::map_t& map, silt::tensor tensor){
  map.uplift = tensor.as<float>();
});

map_t.def_prop_rw("rainfall",
[](soil::map_t& map){
  return silt::tensor(map.rainfall);
},[](soil::map_t& map, silt::tensor tensor){
  map.rainfall = tensor.as<float>();
});

//
// Tracking Fields
//

auto data_t = nb::class_<soil::data_t>(module, "data_t");
data_t.def(nb::init<>());
data_t.def(nb::init<const silt::shape>());

data_t.def_prop_rw("discharge",
  [](soil::data_t& model){
    return silt::tensor(model.discharge);
},[](soil::data_t& model, silt::tensor tensor){
    model.discharge = tensor.as<float>();
});

data_t.def_prop_rw("momentum",
  [](soil::data_t& model){
    return silt::tensor(model.momentum);
},[](soil::data_t& model, silt::tensor tensor){
    model.momentum = tensor.as<float>();
});

data_t.def_prop_rw("mass",
  [](soil::data_t& model){
    return silt::tensor(model.mass);
},[](soil::data_t& model, silt::tensor tensor){
    model.mass = tensor.as<float>();
});

data_t.def_prop_rw("debris",
  [](soil::data_t& model){
    return silt::tensor(model.debris);
},[](soil::data_t& model, silt::tensor tensor){
    model.debris = tensor.as<float>();
});

data_t.def_prop_rw("debris_momentum",
  [](soil::data_t& model){
    return silt::tensor(model.debris_momentum);
},[](soil::data_t& model, silt::tensor tensor){
    model.debris_momentum = tensor.as<float>();
});

//module.def("erode", soil::erode);
*/

// note: consider how to implement this deferred using the nodes
// direct computation? immediate evaluation...

nb::enum_<soil::edge_t>(module, "edge")
  .value("d4",  soil::edge_t::D4)
  .value("d8",  soil::edge_t::D8)
  .export_values();

// module.def("direction", [](const silt::tensor& height){
//   return silt::tensor(soil::direction(height.as<float>()));
// });

module.def("direction", [](const silt::tensor& height, const soil::edge_t edge){
  return silt::tensor(soil::direction(height.as<float>(), edge));
});

module.def("slope", [](const silt::tensor& tensor, const silt::tensor& flow, const silt::vec2 scale){
  return silt::tensor(soil::slope(tensor.as<float>(), flow.as<int>(), scale));
});

// module.def("steepest", [](const silt::tensor& height){
//   return silt::tensor(soil::steepest(height.as<float>()));
// });

module.def("steepest", [](const silt::tensor& height, const soil::edge_t edge){
  return silt::tensor(soil::steepest(height.as<float>(), edge));
});

module.def("random_weighted", [](const silt::tensor& height, const soil::edge_t edge, const size_t seed, const size_t offset, const float T){
  return silt::tensor(soil::random_weighted(height.as<float>(), edge, seed, offset, T));
});

// module.def("accumulate", [](const silt::tensor& graph, const silt::tensor& field){
//   return silt::tensor(soil::accumulate(graph.as<int>(), field.as<float>()));
// });

module.def("accumulate", [](const silt::tensor& graph, const silt::tensor& field, const soil::edge_t edge){
  return silt::tensor(soil::accumulate(graph.as<int>(), field.as<float>(), edge));
});

module.def("gaussian_blur", [](const silt::tensor& tensor, const float sigma){
    return silt::tensor(soil::gaussian_blur(tensor.as<float>(), sigma));
});

module.def("gradient", [](const silt::tensor& tensor, const silt::vec2 scale){
    return silt::tensor(soil::gradient(tensor.as<float>(), scale));
});

module.def("laplacian", [](const silt::tensor& tensor, const silt::vec2 scale){
    return silt::tensor(soil::laplacian(tensor.as<float>(), scale));
});

module.def("negslope", [](const silt::tensor& tensor, const silt::vec2 scale){
    return silt::tensor(soil::negslope(tensor.as<float>(), scale));
});

//
// Path Integral Solution Functions
//

module.def("solve_uniform", [](
  const silt::tensor flow,
  const silt::tensor source,
  const silt::tensor decay,
  silt::tensor rng,
  const silt::vec2 scale,
  const size_t count
){

  return soil::solve_uniform (
    flow.as<float>(),
    source.as<float>(),
    decay.as<float>(),
    rng.as<silt::rng>(),
    scale,
    count
  );

});

// module.def("suspend", [](
//   const silt::tensor flow,
//   const silt::vec2 scale,
//   const float ks
// ){
//   return soil::suspend(flow.as<float>(), scale, ks);
// });

module.def("erode", [](
  silt::tensor height,
  silt::tensor discharge,
  silt::tensor discharge_track,
  silt::tensor mass,
  silt::tensor mass_track,
  silt::tensor momentum,
  silt::tensor momentum_track,
  silt::tensor rng,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
) {
  soil::erode(
    height.as<float>(), 
    discharge.as<float>(),
    discharge_track.as<float>(),
    mass.as<float>(),
    mass_track.as<float>(),
    momentum.as<float>(),
    momentum_track.as<float>(),
    rng.as<silt::rng>(),
    scale, param, mp
  );
});

module.def("erode_debris", [](
  silt::tensor height,
  silt::tensor velocity,
  silt::tensor velocity_track,
  silt::tensor mass,
  silt::tensor mass_track,
  silt::tensor rng,
  const silt::vec3 scale,
  const soil::param_t param,
  const soil::momentum_param_t mp
) {
  soil::erode_debris(
    height.as<float>(), 
    velocity.as<float>(),
    velocity_track.as<float>(),
    mass.as<float>(),
    mass_track.as<float>(),
    rng.as<silt::rng>(),
    scale, param, mp
  );
});

//
// Noise Operations
//

auto noise_t = nb::class_<soil::noise_param_t>(module, "noise_t");
noise_t.def(nb::init<>());
noise_t.def_rw("seed", &soil::noise_param_t::seed);
noise_t.def_rw("gain", &soil::noise_param_t::gain);
noise_t.def_rw("lacunarity", &soil::noise_param_t::lacunarity);
noise_t.def_rw("octaves", &soil::noise_param_t::octaves);
noise_t.def_rw("frequency", &soil::noise_param_t::frequency);
noise_t.def_rw("ext", &soil::noise_param_t::ext);
module.def("noise", &soil::noise);

//
// Masked Operations
//  Note: It would also make sense to just use masks directly...
//  Then we can just have different methods of constructing masks
//  and that would basically let us do any operation within the mask...
//

module.def("masked_set", [](silt::tensor& tensor, const float value, const silt::vec2 center, const float rad){
  soil::masked_set(tensor.as<float>(), value, center, rad);
});

module.def("masked_mean", [](silt::tensor& tensor, const silt::vec2 center, const float rad){
  soil::masked_mean(tensor.as<float>(), center, rad);
});

/*
module.def("flow", [](const silt::tensor& tensor, const soil::index& index){
  return soil::flow(tensor, index);
});

module.def("direction", [](const silt::tensor& tensor, const soil::index& index){
  return soil::direction(tensor, index);
});

module.def("accumulation", [](const silt::tensor& tensor, const soil::index& index, int iterations, int samples){
  return soil::accumulation(tensor, index, iterations, samples);
});

module.def("accumulation_weighted", [](const silt::tensor& tensor, const silt::tensor& weights, const soil::index& index, int iterations, int samples, bool reservoir){
  return soil::accumulation(tensor, weights, index, iterations, samples, reservoir);
});

module.def("accumulation_exhaustive", [](const silt::tensor& tensor, const soil::index& index){
  return soil::accumulation_exhaustive(tensor, index);
});

module.def("accumulation_exhaustive_weighted", [](const silt::tensor& tensor, const soil::index& index, const silt::tensor& weights){
  return soil::accumulation_exhaustive(tensor, index, weights);
});

module.def("upstream", [](const silt::tensor& tensor, const soil::index& index, const glm::ivec2 target){
  return soil::upstream(tensor, index, target);
});

module.def("distance", [](const silt::tensor& tensor, const soil::index& index, const glm::ivec2 target){
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

module.def("sample_lerp", [](const silt::tensor& field, const soil::index& index, const silt::tensor& pos){
  return soil::sample_lerp(field, index, pos);
});

module.def("sample_grad", [](const silt::tensor& field, const soil::index& index, const silt::tensor& pos){
  return soil::sample_grad(field, index, pos);
});

module.def("concat", [](const silt::tensor& a, const silt::tensor& b){
  return soil::concat(a, b);
});

module.def("select_index", [](const silt::tensor& source, const silt::tensor& index){
  return soil::select_index(source, index);
});
*/

}

#endif