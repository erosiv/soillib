#ifndef SOILLIB_OP_COMMON
#define SOILLIB_OP_COMMON

#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <limits>

namespace soil {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1)/thread;
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
// Set Buffer from Value and Buffer
//

template<typename T>
void set_impl(soil::buffer_t<T> buffer, const T val, size_t start, size_t stop, size_t step);

template<typename T>
void set(soil::buffer_t<T> buffer, const T val, size_t start, size_t stop, size_t step){

  if (buffer.host() == soil::host_t::CPU) {
    for(int i = start; i < stop; i += step)
      buffer[i] = val;
  }

  else if (buffer.host() == soil::host_t::GPU) {
    set_impl(buffer, val, start, stop, step);
  }

}

template<typename T>
void set(soil::buffer_t<T> &buffer, const T val) {
  set(buffer, val, 0, buffer.elem(), 1);
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
// Resize Operation
//  Note: Currently only Bilinear Interpolation

template<typename T>
void resize_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs, soil::ivec2 out, soil::ivec2 in);

template<typename T>
void resize(soil::buffer_t<T> &lhs, const soil::buffer_t<T> &rhs, soil::ivec2 out, soil::ivec2 in) {

  if (lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if (lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

  if (lhs.host() == soil::host_t::GPU) {
    resize_impl(lhs, rhs, out, in);
  } else {
    throw soil::error::mismatch_host(soil::host_t::GPU, rhs.host());
  }

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



} // end of namespace soil

#endif