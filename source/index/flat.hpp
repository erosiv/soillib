#ifndef SOILLIB_INDEX_FLAT
#define SOILLIB_INDEX_FLAT

#include <soillib/soillib.hpp>
#include <soillib/util/yield.hpp>
#include <soillib/util/types.hpp>

#include <memory>
#include <array>
#include <vector>
#include <iostream>
#include <variant>

#include <initializer_list>

namespace soil {

//! a shape is a multi-dimensional extent struct.
//! a shape represents a compact N-d region.
//! 
//! a shape is considered static, and doesn't
//! itself implement re-shaping. Instead, it only
//! provides methods to test broadcastability.
//!
//! shape provides a flat index generator, which is
//! iterable and emits the indices for lookup in a
//! hypothetical flat buffer. This takes into account
//! any permutation.
template<size_t D>
struct flat_t {

  typedef glm::vec<D, int> vec_t;

  flat_t() = default;
  flat_t(const std::vector<int>& v){
    for(size_t i = 0; i < v.size(); ++i)
      this->_arr[i] = v[i];
  }

  //! Number of Dimensions
  size_t dims() const { 
    return D; 
  }

  //! Number of Elements
  size_t elem() const {
    size_t v = 1;
    for(size_t d = 0; d < D; ++d)
      v *= this->_arr[d];
    return v;
  }

  //! Shape Dimension Lookup
  size_t operator[](const int d) const {
    if(d >= D) throw std::invalid_argument("index is out of bounds");
    return this->_arr[d];
  }

  //! Position Flattening Procedure
  size_t flat(const vec_t pos) const {
    int value = 0;
    for(size_t d = 0; d < D; ++d){
      value *= this->operator[](d);
      value += pos[d];
    }
    return value;
  }

  //! Out-Of-Bounds Check (Compact)
  bool oob(const vec_t pos) const {
    for(size_t d = 0; d < D; ++d)
      if(pos[d] < 0 || pos[d] >= this->operator[](d)) 
        return true;
    return false;
  }

  //! Position Generator
  //!
  //! This returns a generator coroutine,
  //! which iterates over the set of positions.
  yield<vec_t> iter() const {

    vec_t m_arr{0};   // Initial Condition
    
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
  vec_t _arr;
};

} // end of namespace soil

#endif