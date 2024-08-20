#ifndef SOILLIB_LAYER_CACHED
#define SOILLIB_LAYER_CACHED

#include <soillib/util/types.hpp>
#include <soillib/util/buffer.hpp>

namespace soil {

template<typename T>
struct cached_t: typedbase {

  cached_t(const soil::buffer_t<T> buffer):
    buffer{buffer}{}

  constexpr soil::dtype type() noexcept override { 
    return soil::typedesc<T>::type; 
  }

  T operator()(const size_t index) noexcept {
    return this->buffer[index];
  }

//private:
  soil::buffer_t<T> buffer;
};

struct cached {

  cached(){}

  template<typename T>
  cached(const soil::dtype type, const soil::buffer_t<T> buffer):
    impl{make<T>(type, buffer)}{}

  //! retrieve the strict-typed type enumerator
  inline soil::dtype type() const noexcept {
    return this->impl->type();
  }

  //! check if the contained type is of type
  inline bool is(const soil::dtype type) const noexcept {
    return this->type() == type;
  }

  //! unsafe cast to strict-type
  template<typename T> inline cached_t<T>& as() noexcept {
    return static_cast<cached_t<T>&>(*(this->impl));
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
  static typedbase* make(const soil::dtype type, const soil::buffer_t<T> buffer){
    return typeselect(type, [buffer]<typename S>() -> typedbase* {
      if constexpr (std::same_as<T, S>){
        return new soil::cached_t<S>(buffer);
      } else throw soil::error::cast_error<T, S>{}();
    });
  }

private:
  typedbase* impl;  //!< Strict-Typed Implementation Pointer
};

} // end of namespace soil

#endif