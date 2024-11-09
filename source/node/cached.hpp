#ifndef SOILLIB_LAYER_CACHED
#define SOILLIB_LAYER_CACHED

#include <soillib/core/buffer.hpp>
#include <soillib/core/types.hpp>

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
    return this->buffer.operator[]<T>(index);
  }

  template<typename T>
  T& operator()(const size_t index){ 
    return this->buffer.operator[]<T>(index);
  }

  soil::buffer buffer;
};

} // end of namespace soil

#endif