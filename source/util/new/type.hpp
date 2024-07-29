#ifndef SOILLIB_UTIL_TYPE
#define SOILLIB_UTIL_TYPE

#include <soillib/soillib.hpp>

namespace soil {

struct type {

  // we can probably wrap most of this behavior
  // in some kind of typeinfo structure.
  // this should probably be as constexpr as possible,
  // with static pre-defined types, so that it is easy
  // to construct different polymorphic dynamic containers.

};

}

#endif