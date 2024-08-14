#ifndef SOILLIB_LAYER_CONST
#define SOILLIB_LAYER_CONST

#include <soillib/util/types.hpp>
#include <soillib/util/shape.hpp>
#include <variant>

//! A const layer returns a single value type.
//! ...

namespace soil {

template<typename T>
struct layer_const_t {

  layer_const_t(const T& value):
    value{value}{}

  //! \todo Implement various lookup operators for layer types.
  //! Define what the sensible set is for proper interop, and
  //! implement it across the board for layers.

  T operator()(const size_t& index) const noexcept {
    return this->value;
  }

private:
  const T value;
};

using layer_const_v = std::variant<
  layer_const_t<int>,
  layer_const_t<float>,
  layer_const_t<double>,
  layer_const_t<fvec2>
>;

//! Variant Wrapping Type:
//! Let's us construct different const layer types directly.
//! The type returned to python is a variant.
struct layer_const {

  layer_const(const std::string type, const soil::multi& multi):
    _layer_const{make(type, multi)}{}

  soil::multi operator()(const size_t& index) const {
    return std::visit([&index](auto&& args) -> soil::multi {
      return args(index);
    }, this->_layer_const);
  }

  static layer_const_v make(const std::string type, const soil::multi& multi){
    if(type == "int")     return soil::layer_const_t<int>(std::get<int>(multi));
    if(type == "float")   return soil::layer_const_t<float>(std::get<float>(multi));
    if(type == "double")  return soil::layer_const_t<double>(std::get<double>(multi));
    if(type == "fvec2")   return soil::layer_const_t<fvec2>(std::get<fvec2>(multi));
    throw std::invalid_argument("invalid type argument");
  }

private:
  layer_const_v _layer_const;
};

} // end of namespace soil

#endif