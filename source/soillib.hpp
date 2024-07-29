#ifndef SOILLIB
#define SOILLIB

#include <cstddef>
#include <glm/glm.hpp>
#include <type_traits>
#include <stdexcept>

namespace soil {

template<typename T>
concept matrix_t = requires(T t){
  {float() * t};
};

}; // end of namespace

#endif
