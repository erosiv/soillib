#ifndef SOILLIB_OP_RBF
#define SOILLIB_OP_RBF

#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>
#include <soillib/index/kdtree.hpp>

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
//! rbf can fit data using a kd-tree using various
//! basis function types.
//!
//! rbf supports buffer-based value and gradient sampling.
//!
struct rbf {

  //
  // rbf initialization
  //

  //!< Initialize Centroids from 2D Pointcloud
  void init(const buffer_t<vec2>& centroids);

  size_t elem = 0;          //!< Number of Components
  float shape = 1.0f;       //!< Singular Shape Parameter

  buffer_t<float> weights;  //!< Interpolation Weights
  buffer_t<vec2> centers;   //!< Interpolation Centers

  //
  // Fitting Procedure
  //

  //! Fit the RBF Interpolator to a 2.5D Dataset
  void fit(const kdtree& kdtree, const buffer_t<float>& data, const size_t steps);
  float lrate_w = 0.01f;  //!< Weight Learning Rate

  /*

  //
  // Sampling Methods
  //

  buffer_t<float> sample(const buffer_t<vec2>& pos) const;  //!< Sample the RBF at set of Positions  
  buffer_t<float> sample(const index& index) const;         //!< Sample the RBF for a full Index

  //
  // Basis Function Implementation
  //

  GPU_ENABLE static float func(const float w, const float r, const float s){
    return w / ( 1.0f + (s*s*r*r) );
  }
  
  GPU_ENABLE static float grad_w(const float w, const float r, const float s){
    return 1.0f / ( 1.0f + (s*s*r*r) );
  }

  GPU_ENABLE static float grad_s(const float w, const float r, const float s){
    return - w * 2 * s * r * r / (1.0f + r * r * s * s) / (1.0f + r * r * s * s);
  }

  GPU_ENABLE static vec2 grad_c(const float w, const vec2 d, const float s){
    const float r = glm::length(d);
    return - w * 2 * s * s * r / (1.0f + r * r * s * s) / (1.0f + r * r * s * s) * d;
  }
  */

};

}

#endif