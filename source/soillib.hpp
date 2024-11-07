#ifndef SOILLIB
#define SOILLIB

#include <cstddef>
#include <format>
#include <glm/glm.hpp>
#include <memory>
#include <stdexcept>
#include <type_traits>

#ifndef GPU_ENABLE
#define GPU_ENABLE
#ifdef HAS_CUDA
  #undef GPU_ENABLE
  #define GPU_ENABLE __host__ __device__
#endif
#endif

namespace soil {

template<typename T>
concept matrix_t = requires(T t) {
  { float() * t };
};

}; // namespace soil

#endif
