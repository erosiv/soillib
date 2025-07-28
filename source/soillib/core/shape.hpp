#ifndef SOILLIB_SHAPE
#define SOILLIB_SHAPE

#include <soillib/soillib.hpp>
#include <soillib/core/types.hpp>

namespace soil {

//! shape is a D-dimensional compact extent, with indexing
//! procedures for lookup into linearized tensors and views.
//!
//! shape is intended for 2D and 3D applications, with a D of max 4,
//! for 3D buffers of arbitrary depth.
//! 
//! \todo cleanup this type with cleaner constructors
//! \todo add better flattening / unflattening procedures.
//! \todo consider whether this type should have slice generators.
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

  // note: this should not necessarily be done like this...

  GPU_ENABLE bool oob(const soil::ivec2 pos) const {
    for (size_t d = 0; d < 2; ++d)
      if (pos[d] < 0 || pos[d] >= this->ext[d])
        return true;
    return false;
  }

  GPU_ENABLE bool oob(const soil::ivec3 pos) const {
    for (size_t d = 0; d < 3; ++d)
      if (pos[d] < 0 || pos[d] >= this->ext[d])
        return true;
    return false;
  }
  
  GPU_ENABLE int flatten(const soil::ivec2 pos) const {
    int index{0};
    for (size_t d = 0; d < 2; ++d) {
      index *= this->ext[d];
      index += pos[d];
    }
    return index;
  }

  GPU_ENABLE int flatten(const soil::ivec3 pos) const {
    int index{0};
    for (size_t d = 0; d < 3; ++d) {
      index *= this->ext[d];
      index += pos[d];
    }
    return index;
  }

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

  // Data-Members

  vec_t ext;
  int elem;
  int dim;

};

}

#endif