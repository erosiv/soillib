#ifndef SOILLIB_ERROR
#define SOILLIB_ERROR

#include <soillib/soillib.hpp>
#include <soillib/core/types.hpp>

// Custom Soillib Exceptions

namespace soil {
namespace error {

template<typename From, typename To>
struct cast_error: std::exception {
  static std::string value() noexcept {
    return std::format("invalid cast from <{}> to <{}>", typedesc<From>::name, typedesc<To>::name);
  }
  const char* what() const noexcept override  {
    return value().c_str();
  }
};

} // end of namespace error
} // end of namespace soil

#endif