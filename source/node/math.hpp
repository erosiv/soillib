#ifndef SOILLIB_NODE_COMMON
#define SOILLIB_NODE_COMMON

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <limits>

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
void copy(soil::buffer_t<To>& out, const soil::buffer_t<From>& in, vec2 gmin, vec2 gmax, vec2 gscale, vec2 wmin, vec2 wmax, vec2 wscale, float pscale){

  const ivec2 pmin = ivec2(pscale * (gmin - wmin) / wscale);
  const ivec2 pmax = ivec2(pscale * (gmax - wmin) / wscale);
  const ivec2 pext = ivec2(pscale * (wmax - wmin) / wscale);
  const ivec2 gext = ivec2((gmax - gmin) / gscale);

  for(int x = pmin[1]; x < pmax[1]; ++x){
    for(int y = pmin[0]; y < pmax[0]; ++y){

      const int ind_out = y + pext[0]*(pext[1] - x - 1);

      const size_t px = size_t((pmax[1]-x-1)/pscale);
      const size_t py = size_t((y-pmin[0])/pscale);
      const size_t ind_in = py + px * gext[0];

      out[ind_out] = To(From(pscale) * in[ind_in]);
    }
  }

}

//
// Casting
//

template<typename To, typename From>
soil::buffer_t<To> cast(const soil::buffer_t<From>& buffer){
  
  if (buffer.host() != soil::host_t::CPU)
    throw soil::error::mismatch_host(soil::host_t::CPU, buffer.host());
  
  buffer_t<To> buffer_to(buffer.elem());
  for (auto [i, b] : buffer.const_iter()){
    buffer_to[i] = (To)b;
    //if(!std::isnan(b)){
    //  val = std::min(val, b);
    //}
  }
  return buffer_to;

}

//
// Reductions
//

template<typename T>
T min(const soil::buffer_t<T>& buffer){
  
  if (buffer.host() != soil::host_t::CPU)
    throw soil::error::mismatch_host(soil::host_t::CPU, buffer.host());
  
  T val = std::numeric_limits<T>::max();
  for (auto [i, b] : buffer.const_iter()){
    if(!std::isnan(b)){
      val = std::min(val, b);
    }
  }
  return val;

}

template<typename T>
T max(const soil::buffer_t<T>& buffer){

  if (buffer.host() != soil::host_t::CPU)
    throw soil::error::mismatch_host(soil::host_t::CPU, buffer.host());

  T val = std::numeric_limits<T>::min();
  for (auto [i, b] : buffer.const_iter()){
    if(!std::isnan(b)){
      val = std::max(val, b);
    }
  }
  return val;

}

//
// Set Buffer from Value and Buffer
//

template<typename T>
void set_impl(soil::buffer_t<T> buffer, const T val);

template<typename T>
void set(soil::buffer_t<T> &buffer, const T val) {
  // CPU Implementation
  if (buffer.host() == soil::host_t::CPU) {
    for (auto [i, b] : buffer.iter())
      *b = val;
  }
  // GPU Implementation
  else if (buffer.host() == soil::host_t::GPU) {
    set_impl(buffer, val);
  }
}

template<typename T>
void set_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs);

template<typename T>
void set(soil::buffer_t<T> &lhs, const soil::buffer_t<T> &rhs) {

  if (lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

  if (lhs.host() == soil::host_t::CPU) {
    for (size_t i = 0; i < lhs.elem(); ++i)
      lhs[i] = rhs[i];
  }

  else if (lhs.host() == soil::host_t::GPU) {
    set_impl(lhs, rhs);
  }
}

//
// Resample Buffer using Index
//

template<typename T>
soil::buffer_t<T> resample_impl(const soil::buffer_t<T> &buffer, const soil::index &index);

template<typename T>
soil::buffer_t<T> resample(const soil::buffer_t<T> &input, const soil::index &index) {

  if (input.elem() != index.elem())
    throw soil::error::mismatch_size(input.elem(), index.elem());

  if (input.host() == soil::host_t::CPU) {
    return select(index.type(), [&]<typename I>() {
      auto index_t = index.as<I>();
      soil::flat_t<I::n_dims> flat(index_t.ext());

      soil::buffer_t<T> output(flat.elem());

      using V = soil::typedesc<T>::value_t;
      T value = T{std::numeric_limits<V>::quiet_NaN()};
      set(output, value);

      for (const auto &pos : index_t.iter()) {
        const size_t i = index_t.flatten(pos);
        output[flat.flatten(pos - index_t.min())] = input[i];
      }

      return output;
    });
  }

  else if (input.host() == soil::host_t::GPU) {
    return resample_impl(input, index);
  }

  else
    throw std::invalid_argument("HOST NOT RECOGNIZED");
}

//
// Add Buffer from Buffer and Value In-Place
//

template<typename T>
void add_impl(soil::buffer_t<T> buffer, const T val);

template<typename T>
void add(soil::buffer_t<T> &buffer, const T val) {
  // CPU Implementation
  if (buffer.host() == soil::host_t::CPU) {
    for (size_t i = 0; i < buffer.elem(); ++i)
      buffer[i] += val;
  }
  // GPU Implementation
  else if (buffer.host() == soil::host_t::GPU) {
    add_impl(buffer, val);
  }
}

template<typename T>
void add_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs);

template<typename T>
void add(soil::buffer_t<T> &lhs, const soil::buffer_t<T> &rhs) {

  if (lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

  if (lhs.host() == soil::host_t::CPU) {
    for (size_t i = 0; i < lhs.elem(); ++i)
      lhs[i] += rhs[i];
  }

  else if (lhs.host() == soil::host_t::GPU) {
    add_impl(lhs, rhs);
  }
}

//
// Multiply Buffer from Buffer and Value In-Place
//

template<typename T>
void multiply_impl(soil::buffer_t<T> buffer, const T val);

template<typename T>
void multiply(soil::buffer_t<T> &buffer, const T val) {
  // CPU Implementation
  if (buffer.host() == soil::host_t::CPU) {
    for (size_t i = 0; i < buffer.elem(); ++i)
      buffer[i] *= val;
  }
  // GPU Implementation
  else if (buffer.host() == soil::host_t::GPU) {
    multiply_impl(buffer, val);
  }
}

template<typename T>
void multiply_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs);

template<typename T>
void multiply(soil::buffer_t<T> &lhs, const soil::buffer_t<T> &rhs) {

  if (lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

  if (lhs.host() == soil::host_t::CPU) {
    for (size_t i = 0; i < lhs.elem(); ++i)
      lhs[i] *= rhs[i];
  }

  else if (lhs.host() == soil::host_t::GPU) {
    multiply_impl(lhs, rhs);
  }
}

/*

//
// Invert Buffer
//

template<typename T>
void invert_impl(soil::buffer_t<T> buffer);

template<typename T>
void invert(soil::buffer_t<T>& buffer){
  // CPU Implementation
  if(buffer.host() == soil::host_t::CPU){
    for(size_t i = 0; i < buffer.elem(); ++i)
      buffer[i] = 1.0 / buffer[i];
  }
  // GPU Implementation
  else if(buffer.host() == soil::host_t::GPU){
    invert_impl(buffer);
  }
}
*/

} // namespace soil

#endif
