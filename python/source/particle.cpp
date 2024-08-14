#ifndef SOILLIB_PYTHON_PARTICLE
#define SOILLIB_PYTHON_PARTICLE

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

#include <soillib/particle/particle.hpp>
#include <soillib/particle/water.hpp>

//
// 
//

//! General Layer Binding Function
void bind_particle(py::module& module){

  //auto singular = py::class_<soil::matrix::singular>(module, "singular");
  //singular.def(py::init<>());

  using model_t = soil::water_particle_t;
  auto model = py::class_<model_t>(module, "water_model");
  model.def(py::init<>());
  model.def(py::init<
    soil::shape,
    soil::array,
    soil::constant,
    soil::constant,
    soil::constant
  >());

  model.def_property_readonly("shape", [](model_t& model){ return model.shape; });
  model.def_property_readonly("height", [](model_t& model){ return model.height; });
  model.def_property_readonly("momentum", [](model_t& model){ return model.momentum; });
  model.def_property_readonly("discharge", [](model_t& model){ return model.discharge; });
  model.def_property_readonly("resistance", [](model_t& model){ return model.resistance; });

  using water_t = soil::WaterParticle;
  auto water = py::class_<water_t>(module, "water");
  water.def(py::init<>([](const soil::fvec2 pos){
    return water_t(glm::vec2(pos[0], pos[1]));
  }));

  const soil::WaterParticle_c conf;

  water.def("move", [conf](water_t& water, model_t& model){
    return water.move(model, conf);
  });

  water.def("interact", [conf](water_t& water, model_t& model){
    return water.interact(model, conf);
  });

  water.def_property_readonly("pos", [](water_t& water) -> std::array<float, 2> {
    return {water.pos.x, water.pos.y};
  });
}

#endif