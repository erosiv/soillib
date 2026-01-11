#ifndef SOILLIB_MODEL_PATH_CU
#define SOILLIB_MODEL_PATH_CU
#define HAS_CUDA

#include <soillib/model/path/path.hpp>
#include <silt/core/operation.hpp>
#include <silt/op/common.hpp>
#include <math_constants.h>
#include "sample.hpp"

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

//! Regular Grid / Voxel Traversal Step-Size
//! This function returns the appropriate distance to enter
//! the next voxel's intersection midpoint in a certain direction.
//!
//! Method Outline:
//!  1. Find next X, Y Intersection Distances (Individually)
//!  2. Clamp Distances to a maximum of the Cell-Diagonal
//!  3. Move the mean distance between X and Y intersection
__device__ float __stepsize (
  const silt::vec2 p, //!< Regular Grid Position (Floating Point)
  const silt::vec2 d  //!< Direction (Normalized)
) {

  constexpr float tmax = CUDART_SQRT_TWO_F;

  const float x_neg = __floorf(p.x);
  const float y_neg = __floorf(p.y);
  const float x_pos = 1.0f + x_neg;
  const float y_pos = 1.0f + y_neg;

  const float tx_neg = (x_neg - p.x) / d.x;
  const float tx_pos = (x_pos - p.x) / d.x;
  const float tx = fminf(fmaxf(tx_neg, tx_pos), tmax);

  const float ty_neg = (y_neg - p.y) / d.y;
  const float ty_pos = (y_pos - p.y) / d.y;
  const float ty = fminf(fmaxf(ty_neg, ty_pos), tmax);

  return 0.5f * (tx + ty);

}

template<size_t D>
__global__ void __solve_uniform (
  silt::tensor_t<float> flux,           //!< Flux Integral Estimate [X/s]
  const silt::tensor_t<float> flow,     //!< Flow-Field Tensor      [m/s]
  const silt::tensor_t<float> source,   //!< Source Term Tensor     [X/s]
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor      [1/s]
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source   [1]
  const silt::shape shape,              //!< Tensor Shape
  const silt::vec2 scale,               //!< Cell Scale             [m]
  const float epsilon,                  //!< Minimum Attenuation    [1]
  const float maxstep                   //!< Maximum Particle Steps [1]
) {

  // Note: Number of concurrent samples given
  //  by the dimensionality of the rng tensor.
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= rng.elem()) return;

  // Extract Correct Dimension Views
  using vecD = silt::fvec<D>;               //!< Vector of Dimension
  auto fluxView = flux.view<vecD>();        //!< Tensorised Flux View
  auto sourceView = source.view<vecD>();    //!< Tensorised Source View
  auto flowView = flow.view<silt::vec2>();  //!< Tensorised Flow View

  // Initialize Particle State
  float att = 1.0f;                         //!< Cumulative Flux Attenuation
  float lambda = 0.0f;                      //!< Cumulative Time Travelled
  silt::vec2 pos = silt::vec2 {             //!< Position in Grid-Space
    curand_uniform(&rng[n])*float(shape[0]),
    curand_uniform(&rng[n])*float(shape[1])
  };
  int ind = shape.flatten(pos);             //!< Grid Tensor Index

  // Upstream Contributions Sample
  const float P = 1.0f / float(shape.elem);
  const vecD S = sourceView[ind] / P;
  if(glm::length(S) < epsilon)
    return;

  // the flow evolution rule?
  //  if we don't have any type of momentum, then pits basically
  //  don't really go away. We rely on the well-structuredness of
  //  the velocity field, which we cannot always do.
  //silt::vec2 v = flowView[ind];
  const sample_t<silt::vec2, 2, 1> v_support = sample_t<silt::vec2, 2, 1>::gather(flowView, silt::ivec2(shape[0], shape[1]), pos);
  silt::vec2 v = v_support.val();

  // Integrate along Streamline
  int step = 0;
  while(!shape.oob(pos) && epsilon < abs(att) && ++step < maxstep) {

    // Accumulate Estimate
    const int nind = shape.flatten(pos);  // New Index?
    if(nind != ind) {
      ind = nind;
      if constexpr(D >= 1){
        atomicAdd(&fluxView[ind].x, S.x * att);
      }
      if constexpr(D >= 2){
        atomicAdd(&fluxView[ind].y, S.y * att);
      }
    }

    const sample_t<silt::vec2, 2, 1> v_support = sample_t<silt::vec2, 2, 1>::gather(flowView, silt::ivec2(shape[0], shape[1]), pos);
    v = v_support.val();
//    v = flowView[ind];

    const float v_len = glm::length(v);
    if(v_len < epsilon)
      break;

    // Update the Position in Grid-Space
    const silt::vec2 v_norm = v / v_len;        // Normalized Direction
    const float step = __stepsize(pos, v_norm); // Step-Size in Gridspace
    pos += step * v_norm;                       // Grid Position Update

    // Update the Real Time and Attenuation
    const float dlambda = glm::length(step * scale) / v_len;  // Real Time Travelled
    att *= __expf(-dlambda * decay[ind]);                     // Decay Attenuation
    lambda += dlambda;                                        // Update Real Time

  }

}

template<size_t D>
__global__ void __normalize (
  silt::tensor_t<float> flux,           //!< Flux Integral Estimate []
  const silt::tensor_t<float> flow,     //!< Flow-Field Tensor      []
  const silt::tensor_t<float> source,   //!< Source Term 
  const silt::vec2 scale,               //!< Cell Scale
  const size_t count                    //!< Sample Count
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= flux.elem()/D) return;

  using vecD = silt::fvec<D>;               //!< Vector of Dimension
  auto fluxView = flux.view<vecD>();        //!< Tensorised Flux View
  auto sourceView = source.view<vecD>();    //!< Tensorised Source View
  auto flowView = flow.view<silt::vec2>();  //!< Tensorised Flow View

  const silt::vec2 v = flowView[n];
  const float norm = abs(v.x * scale.y) + abs(v.y * scale.x);
  fluxView[n] = sourceView[n] + fluxView[n] / float(count) / norm;

}

namespace soil {

//! Uniform Distribution Grid-Free Monte-Carlo
//!   Estimator for Linear Conservation Laws
//!
//! Note that this estimator is parameterized so that the 3rd dimension
//! of the source tensor is the dimensionality of the transported quantity.
//!
silt::tensor solve_uniform (
  const silt::tensor_t<float> flow,     //!< Flow-Field Tensor      [m/s x 2]
  const silt::tensor_t<float> source,   //!< Source Term Tensor     [X/s x D]
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor      [1/s x 1]
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source   []
  const silt::vec2 scale,               //!< Cell Scale             [m x 2]
  const size_t count                    //!< Sample Count           []
) {

  // The Accumulated Flux has the same Dimension as the Source-Term.
  //  The entirety of the flux is then initialized to zero.
  //  The actual dimension of the domain doesn't have the third component.
  const silt::shape shapeIn = source.shape();
  const size_t D = shapeIn[2];  // Dimensionality of Transported Quantity
  const silt::shape shape = silt::shape(shapeIn[0], shapeIn[1]);
  auto flux = silt::tensor_t<float>(shapeIn, silt::host_t::GPU);
  silt::set(flux, 0.0f);

  // Simulation Parameters
  const float epsilon = 1E-16f;               //!< Minimum Attenuation [1]
  const float maxstep = shape[0] + shape[1];  //!< Maximum Step Number given by Manhattan Bound

  // Resolve Data-Layout ofw Source / Flux Tensor
  switch(D) {
    case 1:
      __solve_uniform<1><<<block(rng.elem(), 512), 512>>>(flux, flow, source, decay, rng, shape, scale, epsilon, maxstep);
      __normalize<1><<<block(flux.elem(), 512), 512>>>(flux, flow, source, scale, count);
      break;
    case 2:
      __solve_uniform<2><<<block(rng.elem(), 512), 512>>>(flux, flow, source, decay, rng, shape, scale, epsilon, maxstep);
      __normalize<2><<<block(flux.elem(), 512), 512>>>(flux, flow, source, scale, count);
      break;
    default:
      break;
  }

  return silt::tensor(flux);

}

}

#endif