#ifndef SOILLIB_INDEX_FLAT
#define SOILLIB_INDEX_FLAT

#include <soillib/soillib.hpp>
#include <soillib/util/yield.hpp>
#include <soillib/util/types.hpp>

#include <vector>

namespace soil {

namespace {

template<size_t D>
size_t prod(const glm::vec<D, int> vec){
  switch(D){
    case 1: return vec[0];
    case 2: return vec[0]*vec[1];
    case 3: return vec[0]*vec[1]*vec[2];
    case 4: return vec[0]*vec[1]*vec[2]*vec[3];
    default: throw std::invalid_argument("can't flatten vector of dimension > 4");
  }
}

}

//! flat_t<D> is a D-dimensional compact extent
//!
//! this structure allows for conversion between
//! D-dimensional positions and flat indices,
//! with bounds and domain checking.
//!
//! the domain of this index type is compact.
//!
template<size_t D>
struct flat_t: indexbase {

  static_assert(D > 0, "dimension D must be greater than 0");
  static_assert(D <= 4, "dimension D must be less than or equal to 4");
  typedef glm::vec<D, int> vec_t;

  flat_t() = default;
  flat_t(const vec_t _vec):
    _vec{_vec}{}

  //! Number of Dimensions
  constexpr inline size_t dims() const noexcept { 
    return D; 
  }
  
  constexpr soil::dindex type() noexcept override {
    return soil::dindex::FLAT2;
  }

  //! Number of Elements
  inline size_t elem() const {
    return prod<D>(this->_vec);
  }

  inline vec_t min() const {
    return vec_t{0};
  }

  inline vec_t max() const {
    return this->_vec;
  }

  //! Extent Subscript Operator
  inline size_t operator[](const size_t d) const {
    return this->_vec[d];
  }

  // Flattening / Unflattening

  size_t flatten(const vec_t pos) const {
    int value{0};
    for(size_t d = 0; d < D; ++d){
      value *= this->operator[](d);
      value += pos[d];
    }
    return value;
  }

  vec_t unflatten(const int index) const {
    vec_t value{0};
    int scale = 1;
    for(int d = D-1; d >= 0; --d){
      value[d] = ( index / scale ) % this->operator[](d);
      scale *= this->operator[](d);
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
    for(size_t i = 0; i < this->elem(); ++i)
      co_yield unflatten(i);
    co_return;
  }

private:
  vec_t _vec;
};

} // end of namespace soil

#endif