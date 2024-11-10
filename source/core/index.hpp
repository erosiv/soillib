#ifndef SOILLIB_INDEX
#define SOILLIB_INDEX

#include <soillib/index/flat.hpp>
#include <soillib/index/quad.hpp>
#include <iostream>

//! index is a polymorphic index_t wrapper
//!
//! An index allows for the conversion to and from a position type
//! to a flat buffer index, effectively acting as a lookup map.
//!
//! An index is thereby re-usable and gives a scheme for a domain.
//!
//! \todo Determine how this works with layer maps.

namespace soil {

template<typename F, typename... Args>
auto select(const soil::dindex type, F lambda, Args &&...args) {
  switch (type) {
  case soil::dindex::FLAT1:
    return lambda.template operator()<soil::flat_t<1>>(std::forward<Args>(args)...);
  case soil::dindex::FLAT2:
    return lambda.template operator()<soil::flat_t<2>>(std::forward<Args>(args)...);
  case soil::dindex::FLAT3:
    return lambda.template operator()<soil::flat_t<3>>(std::forward<Args>(args)...);
  case soil::dindex::FLAT4:
    return lambda.template operator()<soil::flat_t<4>>(std::forward<Args>(args)...);
  case soil::dindex::QUAD:
    return lambda.template operator()<soil::quad>(std::forward<Args>(args)...);
  default:
    throw std::invalid_argument("index not supported");
  }
}

// select the index type and blablabla do the thing...

struct index {

  template<size_t D>
  using vec_t = glm::vec<D, int>;

  index(){}

  //! \todo remove this template for something better.
  index(const vec_t<1> vec) { this->impl = std::make_shared<flat_t<1>>(vec); }
  index(const vec_t<2> vec) { this->impl = std::make_shared<flat_t<2>>(vec); }
  index(const vec_t<3> vec) { this->impl = std::make_shared<flat_t<3>>(vec); }
  index(const vec_t<4> vec) { this->impl = std::make_shared<flat_t<4>>(vec); }

  index(const std::vector<std::tuple<vec_t<2>, vec_t<2>>> &data) {
    std::vector<quad_node> nodes;
    for (auto &[min, max] : data)
      nodes.emplace_back(min, max);
    this->impl = std::make_shared<quad>(nodes);
  }

  dindex type() const noexcept {
    return this->impl->type();
  }

  template<typename T>
  inline T &as() noexcept {
    return static_cast<T &>(*(this->impl));
  }

  template<typename T>
  inline const T &as() const noexcept {
    return static_cast<T &>(*(this->impl));
  }

  template<size_t D>
  bool oob(const vec_t<D> vec) const {
    return select(impl->type(), [self = this, vec]<typename T>() -> bool {
      if constexpr (std::same_as<vec_t<D>, typename T::vec_t>) {
        auto index = self->as<T>();
        return index.oob(vec);
      } else {
        throw std::invalid_argument("underlying index type does not accept this argument");
      }
    });
  }

  template<size_t D>
  size_t flatten(const vec_t<D> vec) const {
    return select(impl->type(), [self = this, vec]<typename T>() -> size_t {
      if constexpr (std::same_as<vec_t<D>, typename T::vec_t>) {
        auto index = self->as<T>();
        return index.flatten(vec);
      } else {
        throw std::invalid_argument("underlying index type does not accept this argument");
      }
    });
  }

  size_t dims() const {
    return select(impl->type(), [self = this]<typename T>() {
      return self->as<T>().dims();
    });
  }

  size_t elem() const {
    return select(impl->type(), [self = this]<typename T>() {
      return self->as<T>().elem();
    });
  }

  //! \todo eliminate this because it makes no sense
  size_t operator[](const size_t d) const {
    return select(impl->type(), [self = this, d]<typename T>() {
      return self->as<T>().operator[](d);
    });
  }

private:
  using ptr_t = std::shared_ptr<indexbase>;
  ptr_t impl; //!< Strict-Typed Implementation Base Pointer
};

} // end of namespace soil

#endif