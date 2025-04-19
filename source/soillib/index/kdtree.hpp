#ifndef SOILLIB_INDEX_KDTREE
#define SOILLIB_INDEX_KDTREE

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

// #include <cukd/builder.h>
// #include <cukd/knn.h>

namespace soil {

struct payload_t {
  float3 p;
  size_t i;
};

//! kdtree is a wrapper for a kdtree type
//!
//! the type is built so that it works by returning indices,
//! which can be used to then index separate buffers.
struct kdtree {

  kdtree(const soil::buffer& buffer){
      this->allocate(buffer.elem());
      this->setup(buffer.as<vec3>());
  }

  size_t elem() const { return this->_elem; }

  //
  // Allocate / Deallocate
  //

  void allocate(const size_t elem); //!< Allocate Data
  void deallocate();                //!< Deallocate Data

  void setup(const buffer_t<vec3>& buffer);

  buffer_t<int> knnq(const buffer_t<vec3>& query, const size_t k) const;

//  void query()

  buffer knn(const buffer& query, const size_t k) const {
    return soil::buffer(this->knnq(query.as<vec3>(), k));  
  }

private:
  size_t _elem = 0;
  payload_t* data = NULL;
};

}

#endif