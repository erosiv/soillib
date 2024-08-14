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
struct shape_t {

  typedef std::array<size_t, D> arr_t;

  shape_t() = default;
  shape_t(const std::vector<size_t>& v){
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
  size_t operator[](const size_t d) const {
    if(d >= D) throw std::invalid_argument("index is out of bounds");
    return this->_arr[d];
  }

  //! Position Flattening Procedure
  size_t flat(const arr_t pos) const {
    size_t value = 0;
    for(size_t d = 0; d < D; ++d){
      value *= this->operator[](d);
      value += pos[d];
    }
    return value;
  }

  //! Out-Of-Bounds Check (Compact)
  bool oob(const arr_t pos) const {
    for(size_t d = 0; d < D; ++d)
      if(pos[d] < 0 || pos[d] >= this->operator[](d)) 
        return true;
    return false;
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

using shape_v = std::variant<
  soil::shape_t<1>,
  soil::shape_t<2>,
  soil::shape_t<3>
>;

using shape_iter_v = std::variant<
  yield<soil::shape_t<1>::arr_t>,
  yield<soil::shape_t<2>::arr_t>,
  yield<soil::shape_t<3>::arr_t>
>;

// helper type for the visitor #4
template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };


//! Merged Interface Shape Type
struct shape {

  shape(){}
  shape(const std::vector<size_t>& v):
    _shape(make(v)){}

  // Forwarding Implementations

  size_t dims() const {
    return std::visit([](auto&& args){
      return args.dims();
    }, this->_shape);
  }

  size_t elem() const {
    return std::visit([](auto&& args){
      return args.elem();
    }, this->_shape);
  }

  size_t operator[](const size_t d) const {
    return std::visit([&d](auto&& args){
      return args[d];
    }, this->_shape);
  }

  shape_iter_v iter(){
    return std::visit([](auto&& args) -> shape_iter_v {
      return args.iter();
    }, this->_shape);
  }

  // Templated Functions

  template<size_t N>
  size_t flat(const shape_t<N>::arr_t& arr) const {

    if(N != this->dims())
      throw std::invalid_argument("invalid flattening size");
    
    auto tmp = std::get<shape_t<N>>(this->_shape);
    return tmp.flat(arr);
  
  }

  template<size_t N>
  size_t oob(const shape_t<N>::arr_t& arr) const {

    if(N != this->dims())
      throw std::invalid_argument("invalid flattening size");

    auto tmp = std::get<shape_t<N>>(this->_shape);
    return tmp.oob(arr);

  }

  size_t oob(glm::ivec2 pos) const {
    return this->oob<2>(shape_t<2>::arr_t{(size_t)pos.x, (size_t)pos.y});
  }

  size_t flat(glm::ivec2 pos) const {
    return this->flat<2>(shape_t<2>::arr_t{(size_t)pos.x, (size_t)pos.y});
  }

  shape_v make(const std::vector<size_t>& v){
    if(v.size() == 1) return shape_t<1>(v);
    if(v.size() == 2) return shape_t<2>(v);
    if(v.size() == 3) return shape_t<3>(v);
    throw std::invalid_argument("invalid shape size");
  }

  shape_v _shape;
};

} // end of namespace soil

#endif