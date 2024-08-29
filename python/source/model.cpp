#ifndef SOILLIB_PYTHON_MODEL
#define SOILLIB_PYTHON_MODEL

#include <nanobind/nanobind.h>
namespace nb = nanobind;

#include <soillib/core/model.hpp>

#include "glm.hpp"

//
//
//

//! General Layer Binding Function
void bind_model(nb::module_& module){

  //
  // Enumerator
  //

  nb::enum_<soil::component>(module, "dcomp")
  .value("height", soil::component::HEIGHT)
  .value("momentum", soil::component::MOMENTUM)
  .value("discharge", soil::component::DISCHARGE)
  .value("momentum_track", soil::component::MOMENTUM_TRACK)
  .value("discharge_track", soil::component::DISCHARGE_TRACK)
  .value("resistance", soil::component::RESISTANCE)
  .value("maxdiff", soil::component::MAXDIFF)
  .value("settling", soil::component::SETTLING)
  .export_values();

  //
  // 
  //

  auto model = nb::class_<soil::model>(module, "model");
  model.def(nb::init<>());

  model.def_prop_rw("index", 
  [](const soil::model& model) -> soil::index {
    return model.index;
  },
  [](soil::model& model, const soil::index& index){
    model.index = index;
  });

  model.def("__getitem__", [](soil::model& model, const soil::component comp){
    return model[comp];
  }, nb::rv_policy::reference);

  model.def("__setitem__", [](soil::model& model, const soil::component comp, const soil::node node){
    model.set(comp, node);
  });

}

#endif