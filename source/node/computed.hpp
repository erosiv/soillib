#ifndef SOILLIB_LAYER_COMPUTED
#define SOILLIB_LAYER_COMPUTED

#include <soillib/core/error.hpp>
#include <soillib/core/types.hpp>
#include <functional>

namespace soil {

//! computed_t is a strict-typed layer,
//! which has a functor state which is
//! evaluated when indexed.
//!
//! This type is used to implement complex
//! layer types, including coupled layers.
//!
template<typename T>
struct computed_t: typedbase {

  typedef std::function<T(const size_t)> func_t;

  computed_t(func_t func):
    func(func){}

  constexpr soil::dtype type() noexcept override { 
    return soil::typedesc<T>::type; 
  }

  T operator()(const size_t index) noexcept {
    return this->func(index);
  }

private:
  func_t func;
};

//! computed is a dynamically typed computed layer.
//!
struct computed {

  template<typename T>
  using func_t = std::function<T(const size_t)>;

  computed(){}

  template<typename T>
  computed(const soil::dtype type, func_t<T> func):
    impl{make<T>(type, func)}{}
    
  //! retrieve the strict-typed type enumerator
  inline soil::dtype type() const noexcept {
    return this->impl->type();
  }

  //! check if the contained type is of type
  inline bool is(const soil::dtype type) const noexcept {
    return this->type() == type;
  }

  //! unsafe cast to strict-type
  template<typename T> inline computed_t<T>& as() noexcept {
    return static_cast<computed_t<T>&>(*(this->impl));
  }

  template<typename T>
  T operator()(const size_t index){
    return typeselect(this->type(),
      [self=this, index]<typename S>() -> T {
        if constexpr (std::convertible_to<S, T>){
          return (T)self->as<S>().operator()(index);
        } else throw soil::error::cast_error<S, T>{}();
      }
    );
  }

  template<typename T>
  static typedbase* make(const soil::dtype type, func_t<T> func){
    return typeselect(type, [func]<typename S>() -> typedbase* {
      if constexpr (std::same_as<T, S>){
        return new soil::computed_t<S>(func);
      } else if constexpr (std::convertible_to<T, S>){
        return new soil::computed_t<S>([func](const size_t index){
          return (S)func(index);
        });
      } else throw soil::error::cast_error<T, S>{}();
    });
  }

private:
  typedbase* impl;  //!< Strict-Typed Implementation Pointer
};

} // end of namespace soil

#endif