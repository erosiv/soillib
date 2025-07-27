#ifndef SOILLIB_OP_MATH
#define SOILLIB_OP_MATH

#include <limits>
#include <soillib/core/buffer.hpp>
#include <soillib/core/operation.hpp>

namespace soil {

// Note: This is in principle the minimum set of basic
//  operations needed to perform basic bufferized algebra.
//
// - Copy: Allocate + Set
// - Subtract: Multiply by -1, Add
// - Division: Invert and Multiply
// - Out-Of-Place: Copy first, then In-Place
//
// Additional things which would be interesting is taking
// powers, e.g. roots and exponents, and later functions.
// This will be done at a later time though when this
// interface is cleaned up and made more efficient so
// that it doesn't require hugeamounts of code.
//
//!\todo Inversion requires better handling of vector types,
//! and a concrete decision about handling integer types (i.e. conversion)

//
// Copying...
//

template<typename To, typename From>
void copy(soil::buffer_t<To> &out, const soil::buffer_t<From> &in, vec2 gmin, vec2 gmax, vec2 gscale, vec2 wmin, vec2 wmax, vec2 wscale, float pscale) {

  const ivec2 pmin = ivec2(pscale * (gmin - wmin) / wscale);
  const ivec2 pmax = ivec2(pscale * (gmax - wmin) / wscale);
  const ivec2 pext = ivec2(pscale * (wmax - wmin) / wscale);
  const ivec2 gext = ivec2((gmax - gmin) / gscale);

  for (int x = pmin[1]; x < pmax[1]; ++x) {
    for (int y = pmin[0]; y < pmax[0]; ++y) {

      const int ind_out = y + pext[0] * (pext[1] - x - 1);

      const size_t px = size_t((pmax[1] - x - 1) / pscale);
      const size_t py = size_t((y - pmin[0]) / pscale);
      const size_t ind_in = py + px * gext[0];

      out[ind_out] = To(From(pscale) * in[ind_in]);
    }
  }
}

} // namespace soil

#endif
