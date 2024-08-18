#ifndef SOILLIB_LAYER_CONST
#define SOILLIB_LAYER_CONST

#include <soillib/util/types.hpp>
#include <soillib/layer/layer.hpp>

#include <functional>

namespace soil {

template<typename T>
struct computed_t: typedbase, layer_t<T> {

  typedef std::function<T(const size_t index)> func_t;

  computed_t(func_t func):
    func(func){}

  constexpr soil::dtype type() const noexcept override { 
    return soil::typedesc<T>::type; 
  }

  T operator()(const size_t index) noexcept override {
    return this->func(index);
  }

private:
  func_t func;
};

struct computed {

  

};

/*

//! A const layer returns a single value type.
//! ...
template<typename T>
struct constant_t: layer_t<T>, constant_base {

  constant_t(const T value):
    value{value}{}

  soil::dtype type() const noexcept override { 
    return soil::typedesc<T>::type; 
  }

  T operator()(const size_t index) noexcept override {
    return this->value;
  }

private:
  T value;
};

//! Variant Wrapping Type:
//! Let's us construct different const layer types directly.
//! The type returned to python is a variant.
struct constant {

  constant(){}
  constant(const soil::dtype type, const soil::multi& multi):
    _constant{make(type, multi)}{}

  soil::dtype type() const noexcept {
    return this->_constant->type();
  }

  // safe / unsafe retrieval!

  template<typename T>
  constant_t<T>& as() noexcept {
    return static_cast<constant_t<T>&>(*(this->_constant));
  }

  template<typename T>
  constant_t<T>& get(){
    if(this->_constant->type() != soil::typedesc<T>::type)
      throw std::invalid_argument("type is not the stored type");
    return this->as<T>();
  }

  soil::multi operator()(const size_t index){
    return typeselect(this->type(), [self=this, &index]<typename T>() -> soil::multi {
      return self->as<T>().operator()(index);
    });
  }
  
  //! \todo Make this type of constructor automated somehow / generic / templated
  static constant_base* make(const soil::dtype type, const soil::multi& multi){
    return typeselect(type, [&multi]<typename T>() -> constant_base* {
      return new soil::constant_t<T>(std::get<T>(multi));
    });
  }

private:
  constant_base* _constant;
};
*/

} // end of namespace soil

#endif