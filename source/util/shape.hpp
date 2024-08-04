#ifndef SOILLIB_SHAPE
#define SOILLIB_SHAPE

#include <soillib/soillib.hpp>
#include <soillib/util/yield.hpp>

#include <memory>
#include <array>
#include <vector>
#include <iostream>

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
struct shape_b {

  shape_b() = default;
  virtual ~shape_b() = default;

  // Pointer Mangling Interface

  virtual shape_b* clone() const = 0; //!< Virtual Pointer Clone Function

  // Virtual Interface

  virtual size_t dims() const = 0;                      //!< Number of Dimensions
  virtual size_t elem() const = 0;                      //!< Number of Elements
  virtual size_t operator[](const size_t d) const = 0;  //!< Dimension Extent Lookup
};

//! shape_t is a strict-typed, dimensioned shape type.
//!
//! shape_t implements the different procedures for a
//! fixed number of dimensions D.
//!
template<size_t D>
struct shape_t: shape_b {

  typedef std::array<size_t, D> arr_t;

  // Copy / Move Constructors

  shape_t(shape_t& other):
    _arr{other._arr}{}

  shape_t(const shape_t& other):
    _arr{other._arr}{}

  // Actual Constructors

  template<typename ...Args>
  shape_t(Args&& ...args):
    _arr{args...}{}

  // Pointer Mangling Interface

  shape_t* clone() const override {
    return new shape_t(*this);
  }

  // Virtual Interface Implementation

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
    for(size_t i = 0; i < this->elem(); ++i){

      co_yield m_arr; // Yield Current Value
      ++m_arr[D-1];   // Bump Active Dimension

      // Propagate Overflow

      for(size_t d = D-1; d > 0; d -= 1){

        if(m_arr[d] >= this->_arr[d]){
          m_arr[d] = 0;
          ++m_arr[d-1];
        }

      }

    }
    co_return;
  }

private:
  const arr_t _arr;
};

//! shape is a wrapper-type which contains the base type pointer,
//! to guarantee RAII correctness. Additionally, it provides
//! convenience constructors and and cast operators.
//!
struct shape {

  // Copy / Move Constructors:
  // Shape is intended to be copied, therefore
  // the implementation pointer is cloned for
  // copying and removed for moving.

  shape(shape& other):
    _shape(other._shape->clone()){}

  shape(const shape& other):
    _shape(other._shape->clone()){}

  shape(shape&& other):
    _shape(other._shape){
      other._shape = NULL;
    }

  // Constructors

  shape(std::vector<size_t> v):
    _shape{make(v)}{}

  // Interface Implementation

  inline size_t dims() const { return this->_shape->dims(); }
  inline size_t elem() const { return this->_shape->elem(); }

  template<size_t D> shape_t<D>* as(){
    return dynamic_cast<shape_t<D>*>(this->_shape.get());
  }

  template<size_t D> const shape_t<D>* as() const {
    return dynamic_cast<shape_t<D>*>(this->_shape.get());
  }

  inline size_t operator[](const size_t d) {
    return this->_shape->operator[](d);
  }

  // Casting Operators

  explicit operator std::vector<long int>() {
    std::vector<long int> out;
    for(size_t d = 0; d < this->dims(); ++d)
      out.push_back(this->operator[](d));
    return out;
  }

  explicit operator std::vector<size_t>() {
    std::vector<size_t> out;
    for(size_t d = 0; d < this->dims(); ++d)
      out.push_back(this->operator[](d));
    return out;
  }

  template<size_t D>
  inline size_t flat(soil::shape_t<D>::arr_t arr) const {
    if(this->dims() != D)
      throw std::invalid_argument("mismatching dimensions");
    return this->as<D>()->flat(arr);
  }
  
  // Factory Functions

  static shape_b* make(std::vector<size_t> v){
    if(v.size() == 0) throw std::invalid_argument("vector can't have size 0");
    if(v.size() == 1) return new soil::shape_t<1>({v[0]});
    if(v.size() == 2) return new soil::shape_t<2>({v[0], v[1]});
    if(v.size() == 3) return new soil::shape_t<3>({v[0], v[1], v[2]});
    throw std::invalid_argument("vector has invalid size");
  }

private:
  std::shared_ptr<shape_b> _shape;
};

} // end of namespace soil

#endif