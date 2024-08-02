#ifndef SOILLIB_SHAPE
#define SOILLIB_SHAPE

#include <soillib/soillib.hpp>
#include <soillib/util/yield.hpp>

#include <vector>
#include <iostream>

namespace soil {

template<size_t D> struct shape_t;

//! shape_base is an abstract polymorphic shape type,
//! from which fixed-size shape types are derived.
//!
//! a shape is considered static, and doesn't
//! itself implement re-shaping. Instead, it only
//! provides methods to test broadcastability.
//!
//! shape provides a flat index generator, which is
//! iterable and emits the indices for lookup in a
//! hypothetical flat buffer. This takes into account
//! any permutation.
//!
struct shape_base {

  shape_base() = default;
  virtual ~shape_base() = default;

  // Factory Constructor

  //template<typename ...Args>
  // shape(std::vector<size_t> v){
  //   this = make(v);
  // }

  static shape_base* make(std::vector<size_t> v);

  // Virtual Interface

  virtual size_t dims() const = 0;        //!< Number of Dimensions
  virtual size_t elem() const = 0;        //!< Number of Elements
  virtual yield<size_t> iter() const = 0; //!< Flat Index Generator

  //! Dimension Extent Lookup
  virtual size_t operator[](const size_t d) const = 0;

};

//! shape_t is a strict-typed, dimensioned shape type.
//!
//! shape_t implements the different procedures for a
//! fixed number of dimensions D.
//!
template<size_t D>
struct shape_t: shape_base {

  typedef size_t dim_t[D];

  shape_t() = default;

  template<typename... Args>
  shape_t(Args&&... args):
    _extent(std::forward<Args>(args)...){}

  // 

  size_t dims() const {
    return D;
  }

  //! Number of Elements
  size_t elem() const {
    size_t v = 1;
    for(size_t d = 0; d < D; ++d)
      v *= this->_extent[d];
    return v;
  }

  inline size_t flat(const shape_t mod) const {
    return 0;
    //    return val + mod.val * sub.flat(mod.sub);
    //return this->_extent.flat(mod._extent);
  }

  // 

  //! Shape Dimension Lookup
  size_t operator[](const size_t d) const {
    if(d >= D) 
      throw std::invalid_argument("index is out of bounds");
    return this->_extent[d];
  }

  //! Iterator Generator
  //!
  //! \todo consider whether this should actually emit some kind
  //! of vector type instead, which can then in turn be trivially
  //! flattened by the shape into the correctly ordered index.
  //!
  //! this has the advantage of providing the spatial information.
  //!
  yield<size_t> iter() const {
    for(size_t i = 0; i < this->elem(); ++i){
      co_yield i;
    }
    co_return;
  };

private:
  const dim_t _extent;
};




struct shape {

  shape(std::vector<size_t> v):
    _shape{make(v)}{}

  shape(shape& _shape):shape((std::vector<size_t>)_shape){}

  shape(shape&& _shape):
    _shape(_shape._shape){
      _shape._shape = NULL;
    }

  ~shape(){
    if(this->_shape != NULL)
      delete this->_shape;
  }

  // Interface Implementation

  inline size_t dims() {
    return this->_shape->dims();
  }

  inline size_t elem() {
    return this->_shape->elem();
  }

  inline yield<size_t> iter() {
    return this->_shape->iter();
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
  
  //   virtual size_t dims() const = 0;        //!< Number of Dimensions
  // virtual size_t elem() const = 0;        //!< Number of Elements
  // virtual yield<size_t> iter() const = 0; //!< Flat Index Generator

  //! Dimension Extent Lookup
  //virtual size_t operator[](const size_t d) const = 0;

  // Factory Functions

//  template<typename ...Args>
//  static shape* make(Args&& ...args);
  static shape_base* make(std::vector<size_t> v){

    if(v.size() == 0) throw std::invalid_argument("vector can't have size 0");
    if(v.size() == 1) return new soil::shape_t<1>({v[0]});
    if(v.size() == 2) return new soil::shape_t<2>({v[0], v[1]});
    if(v.size() == 3) return new soil::shape_t<3>({v[0], v[1], v[2]});
    throw std::invalid_argument("vector has invalid size");

  }

private:
  shape_base* _shape;
};

} // end of namespace soil

#endif