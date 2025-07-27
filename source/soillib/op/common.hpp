#ifndef SOILLIB_OP_COMMON
#define SOILLIB_OP_COMMON

#include <limits>
#include <soillib/core/buffer.hpp>
#include <soillib/core/types.hpp>
#include <soillib/core/operation.hpp>

namespace soil {

//
// Common Operation Declarations
//

template<typename T>
void set(buffer_t<T> lhs, const T value);

template<typename T>
void add(buffer_t<T> lhs, const T value);

template<typename T>
void multiply(buffer_t<T> lhs, const T value);

template<typename T>
void set(buffer_t<T> lhs, const buffer_t<T> rhs);

template<typename T>
void add(buffer_t<T> lhs, const buffer_t<T> rhs);

template<typename T>
void multiply(buffer_t<T> lhs, const buffer_t<T> rhs);

template<typename T>
void mix(buffer_t<T> lhs, const buffer_t<T> rhs, const float w);

template<typename T>
void clamp(buffer_t<T> lhs, const T min, const T max);

void seed(buffer_t<curandState>& buf, const size_t seed, const size_t offset);

//
// Other Operations that need cleaning / deprecation...
//

//
// Casting
//

template<typename To, typename From>
soil::buffer_t<To> cast(const soil::buffer_t<From> &buffer) {

  if (buffer.host() != soil::host_t::CPU)
    throw soil::error::mismatch_host(soil::host_t::CPU, buffer.host());

  buffer_t<To> buffer_to(buffer.elem());
  for (auto [i, b] : buffer.const_iter()) {
    buffer_to[i] = (To)b;
    // if(!std::isnan(b)){
    //   val = std::min(val, b);
    // }
  }
  return buffer_to;

}

//
// Set Buffer from Value and Buffer
//

template<typename T>
void set_impl(soil::buffer_t<T> buffer, const T val, size_t start, size_t stop, size_t step);

template<typename T>
void set(soil::buffer_t<T> buffer, const T val, size_t start, size_t stop, size_t step) {

  if (buffer.host() == soil::host_t::CPU) {
    for (int i = start; i < stop; i += step)
      buffer[i] = val;
  }

  else if (buffer.host() == soil::host_t::GPU) {
    set_impl(buffer, val, start, stop, step);
  }
}

//
// Resize Operation
//  Note: Currently only Bilinear Interpolation

template<typename T>
void resize_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs, soil::ivec2 out, soil::ivec2 in);

template<typename T>
void resize(soil::buffer_t<T> &lhs, const soil::buffer_t<T> &rhs, soil::ivec2 out, soil::ivec2 in) {

  //  if (lhs.elem() != rhs.elem())
  //    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

  if (lhs.host() == soil::host_t::GPU) {
    resize_impl(lhs, rhs, out, in);
  } else {
    throw soil::error::mismatch_host(soil::host_t::GPU, rhs.host());
  }
}


//
// Legacy Functions
//! \todo get rid of this...

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

//
// Reductions
//

template<typename T>
T min(const soil::buffer_t<T> &buffer) {

  if (buffer.host() != soil::host_t::CPU)
    throw soil::error::mismatch_host(soil::host_t::CPU, buffer.host());

  T val = std::numeric_limits<T>::max();
  for (auto [i, b] : buffer.const_iter()) {
    if (!std::isnan(b)) {
      val = std::min(val, b);
    }
  }
  return val;
}

template<typename T>
T max(const soil::buffer_t<T> &buffer) {

  if (buffer.host() != soil::host_t::CPU)
    throw soil::error::mismatch_host(soil::host_t::CPU, buffer.host());

  T val = std::numeric_limits<T>::min();
  for (auto [i, b] : buffer.const_iter()) {
    if (!std::isnan(b)) {
      val = std::max(val, b);
    }
  }
  return val;
}


} // end of namespace soil

#endif