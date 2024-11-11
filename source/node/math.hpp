#ifndef SOILLIB_NODE_COMMON
#define SOILLIB_NODE_COMMON

#include <soillib/core/buffer.hpp>

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
// Set Buffer from Value and Buffer
//

template<typename T>
void set_impl(soil::buffer_t<T> buffer, const T val);

template<typename T>
void set(soil::buffer_t<T>& buffer, const T val){
  // CPU Implementation
  if(buffer.host() == soil::host_t::CPU){
    for(auto [i, b]: buffer.iter())
      *b = val;
  }
  // GPU Implementation
  else if(buffer.host() == soil::host_t::GPU){
    set_impl(buffer, val);
  }
}

template<typename T>
void set_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs);

template<typename T>
void set(soil::buffer_t<T>& lhs, const soil::buffer_t<T>& rhs){

  if(lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if(lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

  if(lhs.host() == soil::host_t::CPU){
    for(size_t i = 0; i < lhs.elem(); ++i)
      lhs[i] = rhs[i];
  }

  else if(lhs.host() == soil::host_t::GPU){
    set_impl(lhs, rhs);
  }

}

//
// Add Buffer from Buffer and Value In-Place
//

template<typename T>
void add_impl(soil::buffer_t<T> buffer, const T val);

template<typename T>
void add(soil::buffer_t<T>& buffer, const T val){
  // CPU Implementation
  if(buffer.host() == soil::host_t::CPU){
    for(size_t i = 0; i < buffer.elem(); ++i)
      buffer[i] += val;
  }
  // GPU Implementation
  else if(buffer.host() == soil::host_t::GPU){
    add_impl(buffer, val);
  }
}

template<typename T>
void add_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs);

template<typename T>
void add(soil::buffer_t<T>& lhs, const soil::buffer_t<T>& rhs){

  if(lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if(lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

  if(lhs.host() == soil::host_t::CPU){
    for(size_t i = 0; i < lhs.elem(); ++i)
      lhs[i] += rhs[i];
  }

  else if(lhs.host() == soil::host_t::GPU){
    add_impl(lhs, rhs);
  }

}

//
// Multiply Buffer from Buffer and Value In-Place
//

template<typename T>
void multiply_impl(soil::buffer_t<T> buffer, const T val);

template<typename T>
void multiply(soil::buffer_t<T>& buffer, const T val){
  // CPU Implementation
  if(buffer.host() == soil::host_t::CPU){
    for(size_t i = 0; i < buffer.elem(); ++i)
      buffer[i] *= val;
  }
  // GPU Implementation
  else if(buffer.host() == soil::host_t::GPU){
    multiply_impl(buffer, val);
  }
}

template<typename T>
void multiply_impl(soil::buffer_t<T> lhs, const soil::buffer_t<T> rhs);

template<typename T>
void multiply(soil::buffer_t<T>& lhs, const soil::buffer_t<T>& rhs){

  if(lhs.elem() != rhs.elem())
    throw soil::error::mismatch_size(lhs.elem(), rhs.elem());

  if(lhs.host() != rhs.host())
    throw soil::error::mismatch_host(lhs.host(), rhs.host());

  if(lhs.host() == soil::host_t::CPU){
    for(size_t i = 0; i < lhs.elem(); ++i)
      lhs[i] *= rhs[i];
  }

  else if(lhs.host() == soil::host_t::GPU){
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

}

#endif
