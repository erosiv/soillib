#ifndef SOILLIB_SHAPE
#define SOILLIB_SHAPE

#include <soillib/soillib.hpp>

#include <vector>
#include <iostream>

namespace soil {

namespace {

//! dim_t is a recursive N-dimensional extent template
//!
//! dim_t implements a number of convenient operations
//! for an N-dimensional spatial extent.
//!
template<size_t N> 
struct dim_t {

  //! \todo replace this with a template recursive version
  dim_t(size_t d0):val(d0),sub(d0){}
  dim_t(size_t d0, size_t d1):sub(d0),val(d1){}
  dim_t(size_t d0, size_t d1, size_t d2):sub(d0,d1),val(d2){}

  inline size_t prod() const {
    return val * sub.prod();
  }

  inline size_t& last() {
    return this->val;
  }

  inline size_t flat(dim_t<N> mod) const {
    return val + mod.val * sub.flat(mod.sub);
  }

  inline bool isbit() const {
    return this->sub.isbit();
  }

  // Operator Definitions

  inline size_t operator[](size_t n) const {
    if(n == N-1) return val;
    return sub[n];
  }

  inline bool operator==(const dim_t<N> rhs) const {
    return this->val == rhs.val && this->sub == rhs.sub;
  }

  dim_t<N>& operator++() noexcept {
    ++this->val;
    return *this;
  }

  inline dim_t<N>& operator%=(dim_t<N> mod) {
    // execute the modulo capping!
    if(this->val >= mod.val){
      ++this->sub;
      this->val = 0;
    }
    this->sub %= mod.sub;
    return *this;
  }

protected:
  dim_t<N-1> sub;
  size_t val;
};

template<> struct dim_t<0> {

  dim_t(){}
  dim_t(size_t val){}

  inline bool operator==(const dim_t<0> rhs) const {
    return true;
  }

  inline dim_t<0>& operator%=(dim_t<0> mod) {
    return *this;
  }

  inline dim_t<0>& operator++() {
    this->bit = 1;
    return *this;
  }

  inline bool isbit() const {
    return this->bit;
  }

  inline size_t flat(dim_t<0> mod) const { return 0; }
  inline size_t prod() const { return 1; }
  inline size_t operator[](size_t n) const {
    throw std::invalid_argument("argument out of range");
  }

protected:
  bool bit = 0;
};

}




/*
Shape struct:
- Should allow for complex indexing
- Should allow for safe "re-shaping", i.e. casting as a different shape

Once buffers are given shape, they can then be iterable directly from the shape generator.
The choice of shape basically generates indices which are used for lookup.

Finally the concept of a memory and a virtual memory layer can be introduced.

Basically now I need to test the generation of indices.
How should I best do it? with a generator? with an iterator?

Then I can implement the re-shape mechanism.
*/


















template<size_t D> struct shape_t;

struct shape {

  shape() = default;
  virtual ~shape() = default;

  inline virtual size_t operator[](const size_t d) const;
  inline virtual size_t elem() const;

  static shape* make(size_t a);//{ return shape_t<1>}
  static shape* make(size_t a, size_t b);//{ return shape_t<1>}
  static shape* make(size_t a, size_t b, size_t c);//{ return shape_t<1>}
};

template<size_t D> struct shape_iter_t;

template<size_t D>
struct shape_t: shape {

  static constexpr size_t n_dim = D; //!< Number of Dimensions

  shape_t() = default;
  shape_t(dim_t<D> _extent):
    _extent{_extent}{}

  // 

  //! Total Number of Elements
  size_t elem() const {
    return this->_extent.prod();
  }

  inline size_t flat(const shape_t mod) const {
    return this->_extent.flat(mod._extent);
  }

  // 

  //! Shape Dimension Lookup
  size_t operator[](const size_t d) const {
    return this->_extent[d];
  }

  //

  //! Iterator Generator
  
  shape_iter_t<D> begin(){
    auto max = this->_extent;
    max %= this->_extent;

    return shape_iter_t<D>{ dim_t<D>(0), this->_extent };
  }

  shape_iter_t<D> end(){
    auto max = this->_extent;
    max %= this->_extent;

    return shape_iter_t<D>{ max, this->_extent };
  }

private:
  const dim_t<D> _extent;
};

template<size_t D>
struct shape_iter_t {

  shape_iter_t(dim_t<D> _pos, dim_t<D> _mod):
    _pos{_pos},_mod{_mod}{}

  const shape_iter_t<D>& operator++() noexcept {
    ++_pos;
    _pos %= _mod;
    return *this;
  };

  const bool operator==(const shape_iter_t<D>& other) const noexcept {
    return this->_pos.isbit() == other._pos.isbit();
  //  return this->_pos == other._pos;
    //return this->_pos == other._pos && this->_mod == other._mod;
  };

  const bool operator!=(const shape_iter_t<D>& other) const noexcept {
    return !(*this == other);
  };

  // Operators

  shape_t<D> operator*() noexcept {
    return shape_t<D>{_pos};
  };

private:
  dim_t<D> _pos; //!< Current Position
  dim_t<D> _mod; //!< Mod Position
};


#ifdef SHAPE_IMPL

shape* shape::make(size_t a){
  return new shape_t<1>(dim_t<1>(a));
}

shape* shape::make(size_t a, size_t b){
  return new shape_t<2>(dim_t<2>(a, b));
}

shape* shape::make(size_t a, size_t b, size_t c){
  return new shape_t<3>(dim_t<3>(a, b, c));
}

#endif

} // end of namespace soil

#endif