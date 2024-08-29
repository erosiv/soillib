#ifndef SOILLIB_PYTHON_PARTICLE
#define SOILLIB_PYTHON_PARTICLE

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <soillib/particle/particle.hpp>
#include <soillib/particle/water.hpp>

#include "glm.hpp"

//
// 
//

//! General Layer Binding Function
void bind_particle(nb::module_& module){

  using model_t = soil::model;

  using water_t = soil::WaterParticle;
  auto water = nb::class_<water_t>(module, "water");
  water.def(nb::init<soil::vec2>());

  const soil::WaterParticle_c conf;

  water.def("move", [conf](water_t& water, model_t& model){
    return water.move(model, conf);
  });

  water.def("track", &soil::WaterParticle::track);

  water.def("interact", [conf](water_t& water, model_t& model){
    return water.interact(model, conf);
  });

  water.def_prop_ro("pos", [](water_t& water){
    return water.pos;
  });

  water.def_prop_ro("speed", [](water_t& water){
    return water.speed;
  });

  water.def_prop_ro("volume", [](water_t& water){
    return water.volume;
  });
}

#endif