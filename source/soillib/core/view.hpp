#ifndef SOILLIB_VIEW
#define SOILLIB_VIEW

#include <soillib/soillib.hpp>

namespace soil {

//! view_t is a strict-typed, lightweight data-view for vectorized memory reads.
//!
//! A view_t is intended to be constructed on the fly from another data-type, so
//! that any memory read operations from the view are compiler optimized.
//!
//! The view is non-owning and shares its underlying memory, so that it can be
//! passed cheaply while the owning data-types remaing untouched.
//!
//! A check to see whether the construction of a view is valid should be done at
//! construction time, and not at lookup time, to guarantee performance.
//!
template<typename T>
struct view_t {

  GPU_ENABLE view_t(T* data, const size_t elem, const host_t host):
    data(data),
    elem(elem),
    host(host){}

  const size_t elem;  //!< Total Number of Elements T
  const host_t host;  //!< Compute Device Location

  //! Const Subscript Operator
  GPU_ENABLE T operator[](const size_t index) const noexcept {
    return this->data[index];
  }
  
  //! Non-Const Subscript Operator
  GPU_ENABLE T &operator[](const size_t index) noexcept {
    return this->data[index];
  }

private:
  T* data;  //!< Raw Data Pointer
};

template<typename T>
struct const_view_t {

  GPU_ENABLE const_view_t(const T* data, const size_t elem, const host_t host):
    data(data),
    elem(elem),
    host(host){}

  const size_t elem;  //!< Total Number of Elements T
  const host_t host;  //!< Compute Device Location

  //! Const Subscript Operator
  GPU_ENABLE T operator[](const size_t index) const noexcept {
    return this->data[index];
  }

private:
  const T* data;  //!< Raw Data Pointer
};

}

#endif