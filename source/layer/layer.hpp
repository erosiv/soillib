#ifndef SOILLIB_LAYER
#define SOILLIB_LAYER

#include <soillib/soillib.hpp>
#include <soillib/util/types.hpp>

#include <soillib/layer/cached.hpp>
#include <soillib/layer/computed.hpp>
#include <soillib/layer/constant.hpp>

// #include <soillib/layer/stored.hpp>

namespace soil {

//! A layer represents constant, stored or computed
//! quantity distributed over the domain.
//!
//! Layers are modular and can be composed to yield
//! user-defined quantities, which are necessary for
//! executing any dynamic transport model.
//! 
//! Layers can also cache and pre-compute results for
//! efficient deferred computation.

using layer_v = std::variant<
  soil::cached,
  soil::constant,
  soil::computed
>;

struct layer {

  layer(){}

  layer(soil::cached&& _layer):
    _layer{_layer}{}

  layer(soil::constant&& _layer):
    _layer{_layer}{}

  layer(soil::computed&& _layer):
    _layer{_layer}{}

  layer(layer_v&& _layer):
    _layer{_layer}{}

  template<typename T>
  T operator()(const size_t index) {
    return std::visit([index](auto&& args){
      return args.template operator()<T>(index);
    }, this->_layer);
  }

  soil::dtype type() const {
    return std::visit([](auto&& args){
      return args.type();
    }, this->_layer);
  }

  layer_v _layer;
};

} // end of namespace soil

#endif