#ifndef SOILLIB
#define SOILLIB

#include <cstddef>
#include <format>
#include <glm/glm.hpp>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace soil {

template<typename T>
concept matrix_t = requires(T t) {
  { float() * t };
};

}; // namespace soil

#endif
