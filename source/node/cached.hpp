#ifndef SOILLIB_LAYER_CACHED
#define SOILLIB_LAYER_CACHED

#include <soillib/core/buffer.hpp>
#include <soillib/core/types.hpp>
#include <soillib/util/error.hpp>

namespace soil {

struct cached: nodebase {

  cached() {}
  cached(const soil::buffer buffer): buffer{buffer} {}

  constexpr soil::dnode node() noexcept override {
    return soil::CACHED;
  }

  //! retrieve the strict-typed type enumerator
  inline soil::dtype type() const noexcept {
    return this->buffer.type();
  }

  //! check if the contained type is of type
  inline bool is(const soil::dtype type) const noexcept {
    return this->type() == type;
  }

  //! templated lookup operator (cast to T)
  //!
  //! Note that this performs a cast to the desired type.
  //! A static check is performed to guarantee that the
  //! cast of the actual internal type is valid.
  template<typename T>
  T operator()(const size_t index) const {
    return select(this->buffer.type(), [self = this, index]<typename S>() -> T {
      if constexpr (std::same_as<T, S>) {
        return self->buffer.as<S>().operator[](index);
      } else if constexpr (std::convertible_to<S, T>) {
        return (T)self->buffer.as<S>().operator[](index);
      } else
        throw soil::error::cast_error<S, T>();
    });
  }

  soil::buffer buffer;
};

} // end of namespace soil

#endif