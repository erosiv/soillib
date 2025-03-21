#ifndef SOILLIB_OP_RBF
#define SOILLIB_OP_RBF

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

// Radial Basis Function Interpolator
// 1. Fit the Weight-Set from a set of Values, Positions
//  1.1 For Different Interpolator Types
//  1.2 Allow for Shape-Parameter Optimization as well?
// 2. Compute the Values from a Weight-Set, Positions
//  Kernelize this for Performance
//  Obviously, implement the gradient of this as well.

namespace soil {

//! rbf is a radial basis function interpolator
//!
//! rbf supports various basis function types,
//! and provides host and device functions for sampling.
//!
struct rbf {

  //! Note: Make differentiable for Gradient-Descent
  GPU_ENABLE static float func(const float dist, const float shape){
    return 1.0f / ( 1.0f + (shape * dist) * (shape * dist) );
  }

  buffer_t<float> sample(const buffer_t<vec2>& pos) const;  //!< Sample the RBF at set of Positions  
  buffer_t<float> sample(const index& index) const;         //!< Sample the RBF for a full Index

  //! Fit the RBF Interpolator to a 2.5D Dataset
  //! Note: This can be split into two buffers in theory.
  void fit(const buffer_t<vec3>& data, const size_t steps);

private:
  buffer_t<float> weights;  //!< Interpolation Weights
  buffer_t<vec2> pos;       //!< Positions
};

}

#endif