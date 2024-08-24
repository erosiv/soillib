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

  //auto singular = nb::class_<soil::matrix::singular>(module, "singular");
  //singular.def(nb::init<>());

  using model_t = soil::water_particle_t;
  auto model = nb::class_<model_t>(module, "water_model");
  model.def(nb::init<>());
  model.def(nb::init<
    soil::index,
    soil::node,
    soil::node,
    soil::node,
    soil::node,
    soil::node,
    soil::node,
    soil::node,
    soil::node
  >());

  model.def_prop_ro("index", [](model_t& model){ return model.index; });
  model.def_prop_ro("height", [](model_t& model){ return model.height; });
  model.def_prop_ro("momentum", [](model_t& model){ return model.momentum; });
  model.def_prop_ro("discharge", [](model_t& model){ return model.discharge; });
  model.def_prop_ro("resistance", [](model_t& model){ return model.resistance; });
  model.def_prop_ro("maxdiff", [](model_t& model){ return model.maxdiff; });
  model.def_prop_ro("settling", [](model_t& model){ return model.settling; });

  model.def_prop_ro("momentum_track", [](model_t& model){ return model.momentum_track; });
  model.def_prop_ro("discharge_track", [](model_t& model){ return model.discharge_track; });

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