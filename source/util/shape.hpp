#ifndef SOILLIB_SHAPE
#define SOILLIB_SHAPE

#include <soillib/soillib.hpp>
#include <soillib/util/yield.hpp>

#include <memory>
#include <array>
#include <vector>
#include <iostream>
#include <variant>

#include <initializer_list>

namespace soil {

// a shape is a multi-dimensional extent struct.
// 
// a shape is considered static, and doesn't
// itself implement re-shaping. Instead, it only
// provides methods to test broadcastability.
//
// shape provides a flat index generator, which is
// iterable and emits the indices for lookup in a
// hypothetical flat buffer. This takes into account
// any permutation.

//! shape_b is an polymorphic shape base type,
//! from which fixed-size shape types are derived.
//!
//! shape_t is a strict-typed, dimensioned shape type.
//!
//! shape_t implements the different procedures for a
//! fixed number of dimensions D.
//!
template<size_t D>
struct shape_t {

  typedef std::array<size_t, D> arr_t;

  shape_t() = default;
  shape_t(const std::vector<size_t>& v){
    for(size_t i = 0; i < v.size(); ++i)
      this->_arr[i] = v[i];
  }

  size_t dims() const { return D; }

  //! Number of Elements
  size_t elem() const {
    size_t v = 1;
    for(size_t d = 0; d < D; ++d)
      v *= this->_arr[d];
    return v;
  }

  inline size_t flat(const arr_t pos) const {
    size_t value = 0;
    for(size_t d = 0; d < D; ++d){
      value *= this->operator[](d);
      value += pos[d];
    }
    return value;
  }

  //! Shape Dimension Lookup
  size_t operator[](const size_t d) const {
    if(d >= D) throw std::invalid_argument("index is out of bounds");
    return this->_arr[d];
  }

  //! Position Generator
  //!
  //! This returns a generator coroutine,
  //! which iterates over the set of positions.
  yield<arr_t> iter() const {

    arr_t m_arr{0};   // Initial Condition
    
    // Generate Values: One Per Element!
    const size_t elem = this->elem();
    for(size_t i = 0; i < elem; ++i){
      
      co_yield m_arr; // Yield Current Value
      ++m_arr[D-1];   // Bump Active Dimension

      // Propagate Overflow

      for(size_t d = D-1; d > 0; d -= 1){

        if(m_arr[d] >= this->_arr[d]){
          m_arr[d] = 0;
          ++m_arr[d-1];
        }
        else break;

      }

    }
    co_return;
  }

private:
  arr_t _arr;
};

using shape = std::variant<
  soil::shape_t<1>,
  soil::shape_t<2>,
  soil::shape_t<3>
>;

} // end of namespace soil

#endif