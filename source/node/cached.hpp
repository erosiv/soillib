#ifndef SOILLIB_LAYER_CACHED
#define SOILLIB_LAYER_CACHED

#include <soillib/core/buffer.hpp>
#include <soillib/core/types.hpp>
#include <soillib/util/error.hpp>

namespace soil {

template<typename T>
struct cached_t: typedbase {

  cached_t(const soil::buffer_t<T> buffer): buffer{buffer} {}

  constexpr soil::dtype type() noexcept override {
    return soil::typedesc<T>::type;
  }

  T operator()(const size_t index) const noexcept {
    return this->buffer[index];
  }

  T &operator()(const size_t index) noexcept {
    return this->buffer[index];
  }

  // private:
  soil::buffer_t<T> buffer;
};

struct cached: nodebase {

  cached() {}
  cached(const soil::buffer buffer): impl{make(buffer)} {}

  constexpr soil::dnode node() noexcept override {
    return soil::CACHED;
  }

  //! retrieve the strict-typed type enumerator
  inline soil::dtype type() const noexcept {
    return this->impl->type();
  }

  //! check if the contained type is of type
  inline bool is(const soil::dtype type) const noexcept {
    return this->type() == type;
  }

  //! unsafe cast to strict-type
  template<typename T>
  inline cached_t<T> &as() noexcept {
    return static_cast<cached_t<T> &>(*(this->impl));
  }

  template<typename T>
  inline const cached_t<T> &as() const noexcept {
    return static_cast<cached_t<T> &>(*(this->impl));
  }

  //! templated lookup operator (cast to T)
  //!
  //! Note that this performs a cast to the desired type.
  //! A static check is performed to guarantee that the
  //! cast of the actual internal type is valid.
  template<typename T>
  T operator()(const size_t index) {
    return select(this->type(), [self = this, index]<typename S>() -> T {
      if constexpr (std::same_as<T, S>) {
        return self->as<S>().operator()(index);
      } else if constexpr (std::convertible_to<S, T>) {
        return (T)self->as<S>().operator()(index);
      } else
        throw soil::error::cast_error<S, T>();
    });
  }

private:
  using ptr_t = std::shared_ptr<typedbase>;
  ptr_t impl; //!< Strict-Typed Implementation Base Pointer

  static ptr_t make(const soil::buffer buffer) {
    return select(buffer.type(), [&buffer]<typename S>() -> ptr_t {
      soil::buffer_t<S> buffer_t = buffer.as<S>();
      return std::make_shared<soil::cached_t<S>>(buffer_t);
    });
  }
};

} // end of namespace soil

#endif