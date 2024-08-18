#ifndef SOILLIB_LAYER_CONSTANT
#define SOILLIB_LAYER_CONSTANT

#include <soillib/util/types.hpp>

namespace soil {

//! I don't know why it breaks without this.
template<typename T>
struct layer_t {
  //! Compute the Output Quantity
  virtual T operator()(const size_t index) noexcept;
};

//! constant_t is a strict-typed layer,
//! which returns a single value.
template<typename T>
struct constant_t: typedbase, layer_t<T> {

  constant_t(const T value):
    value{value}{}

  constexpr soil::dtype type() noexcept override { 
    return soil::typedesc<T>::type;
  }

  T operator()(const size_t index) noexcept override {
    return this->value;
  }

private:
  const T value;
};

//! constant is a dynamically typed constant_t wrapper.
//!
//! constant can be safely and unsafely cast
//! to a statically known constant_t<T>.
struct constant {

  constant(){}

  template<typename T>
  constant(const soil::dtype type, const T value):
    impl{make<T>(type, value)}{}

  //! retrieve the strict-typed type enumerator
  inline soil::dtype type() const noexcept {
    return this->impl->type();
  }

  //! check if the contained type is of type
  inline bool is(const soil::dtype type) const noexcept {
    return this->type() == type;
  }

  //! unsafe cast to strict-type
  template<typename T> inline constant_t<T>& as() noexcept {
    return static_cast<constant_t<T>&>(*(this->impl));
  }

  //! templated lookup operator (cast to T)
  //!
  //! Note that this performs a cast to the desired type.
  //! A static check is performed to guarantee that the
  //! cast of the actual internal type is valid.
  template<typename T>
  T operator()(const size_t index){
    return typeselect(this->type(),
      [self=this, index]<typename S>() -> T {
        if constexpr (std::same_as<T, S>){
        return self->as<S>().operator()(index);
        } else if constexpr (std::convertible_to<S, T>){
          return (T)self->as<S>().operator()(index);
        } else throw soil::error::cast_error<S, T>{}();
      }
    );
  }

  template<typename T>
  static typedbase* make(const soil::dtype type, const T value){
    return typeselect(type, [value]<typename S>() -> typedbase* {
      if constexpr (std::same_as<T, S>){
        return new soil::constant_t<S>(value);
      } else if constexpr (std::convertible_to<T, S>){
        return new soil::constant_t<S>(S(value));
      } else throw soil::error::cast_error<T, S>{}();
    });
  }

private:
  typedbase* impl;  //!< Strict-Typed Implementation Pointer
};

} // end of namespace soil

#endif