#ifndef SOILLIB_ERROR
#define SOILLIB_ERROR

#include <soillib/soillib.hpp>

namespace soil {
namespace error {

// Error Messages
//! \todo this is exemplary, clean this up.

template<typename From, typename To>
struct cast_error {
  static std::invalid_argument operator()(){
    return std::invalid_argument(value());
  }
  static std::string value(){
    return std::format("invalid cast from <{}> to <{}>", typedesc<From>::name, typedesc<To>::name);
  }
};

} // end of namespace error
} // end of namespace soil

#endif