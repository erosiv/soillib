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

  //! Initialize Centroids from 2D Pointcloud
  void init(const buffer_t<vec2>& centroids);

  GPU_ENABLE inline size_t elem() const {
    return this->_elem;
  }
  
  buffer_t<float> weights;  //!< Interpolation Weights
  buffer_t<vec2> centers;   //!< Interpolation Centers

  float shape = 1.0f;       //!< Singular Shape Parameter
  size_t P = 3;             //!< Polynomial Weights

  //
  // Fitting Procedure
  //  These two tensors form a least-squares system that jointly
  //  solves for the weights of the RBF and monomial basis set.

  buffer_t<float> matrix(const buffer_t<vec2>& samples) const;  //!< RBF Interpolation Matrix
  buffer_t<float> vector(const buffer_t<float>& values) const;  //!< RBF Interoplation Vector

  //
  // Sampling Methods
  //

  buffer_t<float> sample(const buffer_t<vec2>& pos)     const;  //!< Sample Dense RBF w. Sparse Positions
  buffer_t<float> sample(const soil::flat_t<2>& index)  const;  //!< Sample Dense RBF w. Dense Index

  buffer_t<float> sample(const buffer_t<vec2>& pos, const soil::kdtree& kdtree)     const;  //!< Sample Dense RBF w. Sparse Positions
  buffer_t<float> sample(const soil::flat_t<2>& index, const soil::kdtree& kdtree)  const;  //!< Sample Sparse RBF w. Dense Index

  //
  // Basis Function Implementation
  //

  //! Inverse Quadratic
//  __device__ static float func(const float r){
//    return 1.0f / ( 1.0f + r*r );
//  }
  
  //! Gaussian
  __device__ static float func(const float r){
    return __expf(- r * r);
  }

  //! Bump Function
//  __device__ static float func(const float r){
//    if(r > 1.0f) return 0.0f;
//    return expf(- 1.0f / (1.0f - r * r));
//  }

private:
  size_t _elem = 0;
};

}

#endif