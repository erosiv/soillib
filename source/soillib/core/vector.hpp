#ifndef SOILLIB_VECTOR
#define SOILLIB_VECTOR

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

//
// soillib vector type definitions / aliases
//  soillib uses glm as its small vector type implementation.
//  this is causing some portability issues and might be replaced.
//  

namespace soil {

// Vector Type Aliases (Convenience)

constexpr auto defaultp = glm::qualifier::packed_highp;

template<size_t D>
using ivec = glm::vec<D, int, defaultp>;
template<size_t D>
using uvec = glm::vec<D, unsigned int, defaultp>;
template<size_t D>
using fvec = glm::vec<D, float, defaultp>;
template<size_t D>
using dvec = glm::vec<D, double, defaultp>;
template<size_t D>
using vec = fvec<D>; // Default Precision

using ivec1 = ivec<1>;
using ivec2 = ivec<2>;
using ivec3 = ivec<3>;
using ivec4 = ivec<4>;

using uvec1 = uvec<1>;
using uvec2 = uvec<2>;
using uvec3 = uvec<3>;
using uvec4 = uvec<4>;

using fvec1 = fvec<1>;
using fvec2 = fvec<2>;
using fvec3 = fvec<3>;
using fvec4 = fvec<4>;

using dvec1 = dvec<1>;
using dvec2 = dvec<2>;
using dvec3 = dvec<3>;
using dvec4 = dvec<4>;

using vec1 = vec<1>;
using vec2 = vec<2>;
using vec3 = vec<3>;
using vec4 = vec<4>;

} // end of namespace soil

#endif