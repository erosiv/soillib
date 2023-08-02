#ifndef SOILLIB_MAP
#define SOILLIB_MAP

#include <soillib/soillib.hpp>

namespace soil {
namespace map {

template<typename T>
concept map_t = requires(T t){
  // Base Operations 
  { t.oob(glm::ivec2()) } -> std::same_as<bool>;
  { t.get(glm::ivec2()) };
  // Iterators
  { t.begin() };
  { t.end() };
};

}

}

#endif