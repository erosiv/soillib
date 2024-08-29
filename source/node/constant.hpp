#ifndef SOILLIB_LAYER_CONSTANT
#define SOILLIB_LAYER_CONSTANT

#include <soillib/core/types.hpp>
#include <soillib/util/error.hpp>

namespace soil {

//! constant_t is a strict-typed layer,
//! which returns a single value.
template<typename T>
struct constant_t: typedbase {

  constant_t(const T value): value{value} {}

  constexpr soil::dtype type() noexcept {
    return soil::typedesc<T>::type;
  }

  T operator()(const size_t index) const noexcept {
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

  constant() {}

  template<typename T>
  constant(const soil::dtype type, const T value): impl{make<T>(type, value)} {}

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
  inline constant_t<T> &as() noexcept {
    return static_cast<constant_t<T> &>(*(this->impl));
  }

  //! unsafe cast to strict-type
  template<typename T>
  inline const constant_t<T> &as() const noexcept {
    return static_cast<constant_t<T> &>(*(this->impl));
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

  template<typename T>
  static ptr_t make(const soil::dtype type, const T value) {
    return select(type, [value]<typename S>() -> ptr_t {
      if constexpr (std::same_as<T, S>) {
        return std::make_shared<soil::constant_t<S>>(value);
      } else if constexpr (std::convertible_to<T, S>) {
        return std::make_shared<soil::constant_t<S>>(S(value));
      } else
        throw soil::error::cast_error<T, S>();
    });
  }
};
} // end of namespace soil

#endif