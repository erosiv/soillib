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

  // rbf centroid initialization

  void init(const buffer_t<vec3>& data);          //!< Initialize Centroids from Pointcloud
  void init(const index& index, const size_t N);  //!< Initialize Centroids Randomly

  buffer_t<float> sample(const buffer_t<vec2>& pos) const;  //!< Sample the RBF at set of Positions  
  buffer_t<float> sample(const index& index) const;         //!< Sample the RBF for a full Index

  //! Fit the RBF Interpolator to a 2.5D Dataset
  //! Note: This can be split into two buffers in theory.
  void fit(const buffer_t<vec3>& data, const size_t steps);

  // Parameters

  float lrate = 0.01f;
  float shape = 1.0f;

  size_t elem = 0;          //!< Number of Components
  buffer_t<float> weights;  //!< Interpolation Weights
  buffer_t<float> shapes;   //!< Interpolation Shape Parameters
  buffer_t<vec2> centers;   //!< Interpolation Centers

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

};

}

#endif