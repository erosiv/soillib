#ifndef SOILLIB_LAYER_CONST
#define SOILLIB_LAYER_CONST

#include <soillib/util/types.hpp>
#include <soillib/util/shape.hpp>
#include <variant>

//! A const layer returns a single value type.
//! ...

namespace soil {

template<typename T>
struct constant_t {

  constant_t(){}
  constant_t(const T& value):
    value{value}{}

  //! \todo Implement various lookup operators for layer types.
  //! Define what the sensible set is for proper interop, and
  //! implement it across the board for layers.

  T operator()(const size_t& index) const noexcept {
    return this->value;
  }

private:
  T value;
};

using constant_v = soil::multi_t<constant_t>;

//! Variant Wrapping Type:
//! Let's us construct different const layer types directly.
//! The type returned to python is a variant.
struct constant {

  constant(){}
  constant(const std::string type, const soil::multi& multi):
    _constant{make(type, multi)}{}

  soil::multi operator()(const size_t& index) const {
    return std::visit([&index](auto&& args) -> soil::multi {
      return args(index);
    }, this->_constant);
  }

  //! \todo Make this type of constructor automated somehow / generic / templated
  static constant_v make(const std::string type, const soil::multi& multi){
    if(type == "int")     return soil::constant_t<int>(std::get<int>(multi));
    if(type == "float")   return soil::constant_t<float>(std::get<float>(multi));
    if(type == "double")  return soil::constant_t<double>(std::get<double>(multi));
    if(type == "vec2")    return soil::constant_t<vec2>(std::get<vec2>(multi));
    throw std::invalid_argument("invalid type argument");
  }

private:
  constant_v _constant;
};

} // end of namespace soil

#endif