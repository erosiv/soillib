#ifndef SOILLIB_SHAPE
#define SOILLIB_SHAPE

#include <soillib/soillib.hpp>
#include <soillib/core/types.hpp>

// A shape should replace the multitude of flat indices,
//  which only describe the spatial extent of a box tensor.
//  More complicated shapes will not be represented here for now.
//  We can later think about what a dynamic layering structure looks like,
//  but for now we need to just implement shapes for efficiency.
//  This will cover our majority use cases! Which we need for release!!!
//  We can think in particular about what a quad map looks like later,
//  or other irregular shapes. But that's not important ATM.
//  something something bounding box hierarchies etc.

// Beyond shape, we should introduce a tensor type that combines
// the buffer with a shape. Certain operations then work on tensors,
// not on buffers (shaped vs. shapeless operations).

namespace soil {

/*
//! vector type is essential to operation...
//! we note that we can have buffers of a type with a shape,
//! so structured buffers are not the problem per se.
//! what we want instead is to have buffers with vec2 or vec3
//! types at the end without having to maintain those somehow
//! as type multiplexes. That is super annoying. So instead,
//! the last dimension of the buffer then constructs the vec type.
//! we will have to see how we do that final part but it should be
//! possible using slices somehow... something like a cast operator.
template<typename T, size_t N>
struct vector {

  GPU_ENABLE inline int operator[](const size_t d) const {
    return this->data[d];
  }

  // ... add a bunch of operators ...
  // ... but this is basically the core function already ...

private:
  T data[N] = {T(0)};  // Zero Initialization
};
*/

// ... slice type ...
// ... cast to vector ...

//! shape is a D-dimensional compact extent
//! it is an indexing structure that allows for lookup
//! into linearized data buffers.
//!
//! shape is primarily intended for 2D and 3D applications.
//! D is maximally 4, which allows for 3D buffers of arbitrary depth.
//!
struct shape {

  // Constructors

  using vec_t = glm::vec<4, int>;

  GPU_ENABLE shape(int d0, int d1, int d2, int d3):
    ext{d0, d1, d2, d3},
    elem{d0*d1*d2*d3},
    dim{4}{}

  GPU_ENABLE shape(int d0, int d1, int d2):
    ext{d0, d1, d2, 1},
    elem{d0*d1*d2},
    dim{3}{}

  GPU_ENABLE shape(int d0, int d1):
    ext{d0, d1, 1, 1},
    elem{d0*d1},
    dim{2}{}

  GPU_ENABLE shape(int d0):
    ext{d0, 1, 1, 1},
    elem{d0},
    dim{1}{}

  GPU_ENABLE shape():
    ext{1, 1, 1, 1},
    elem{1},
    dim{0}{}

  //! Dimension Subscript Operator
  GPU_ENABLE inline int operator[](const size_t d) const {
    return this->ext[d];
  }

  //! Out-Of-Bounds Check (Compact)
  GPU_ENABLE bool oob(const vec_t pos) const {
    for (size_t d = 0; d < this->dim; ++d)
      if (pos[d] < 0 || pos[d] >= this->ext[d])
        return true;
    return false;
  }

  //
  // Question: What is the ideal way to initialize and flatten / unflatten?
  //  Do we accept vectors of different size and shape? What does a slice do?
  //  Do we accept and return slices that allow for weird iteration patterns?
  //  What kind of vector type do we accept? We basically have to replace glm...
  //  So if we return a slice, it generates every index... and then we can downcast
  //  a slice directly to a vector which populates it with the positions right?
  //  

  /*
  //! Flattening Operator
  GPU_ENABLE int flatten(const vec_t pos) const {
    int index{0};
    for (size_t d = 0; d < this->dim; ++d) {
      index *= this->ext[d];
      index += pos[d];
    }
    return index;
  }
  
  //! Unflattening Operator
  GPU_ENABLE vec_t unflatten(const int index) const {
    vec_t value{0};
    int scale = 1;
    for (int d = this->dim - 1; d >= 0; --d) {
      value[d] = (index / scale) % this->ext[d];
      scale *= this->ext[d];
    }
    return value;
  }
  */

  // Data-Members

  vec_t ext;
  int elem;
  int dim;

};

}

#endif