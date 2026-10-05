#include <soillib/soillib.hpp>

#include <soillib/model/path/path.hpp>
#include <silt/core/operation.hpp>
#include <silt/op/common.hpp>
#include <math_constants.h>
#include "sample.hpp"

namespace {

inline int block(const int64_t elem, const int thread) {
  return int((elem + thread - 1) / thread);
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

  const float x_neg = floorf(p.x);
  const float y_neg = floorf(p.y);
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

//! Terminal velocity of a particle at a position in grid-space: the velocity at
//! which the drag balances the acceleration down the potential gradient. This
//! is the velocity the motion law relaxes toward, and the whole of the velocity
//! field the solver needs -- nothing is prescribed from outside.
__device__ silt::vec2 __velocity (
  const silt::view_t<const float>& height, //!< Height Field          [1]
  const silt::ivec2 res,                   //!< Field Resolution      [1]
  const silt::vec2 pos,                    //!< Grid-Space Position   [1]
  const silt::vec3 scale,                  //!< Cell Scale            [m]
  const float gravity,                     //!< Gravity               [m/time^2]
  const float drag                         //!< Drag Rate             [1/time]
) {

  // Height is in normalized units, so the per-cell difference becomes a
  // dimensionless slope through scale.z / scale.xy -- the same convention the
  // soil::grad_* operators use.
  const auto support = sample_t<float, 2, 1>::gather(height, res, pos);
  const silt::vec2 grad = support.grad();
  const silt::vec2 slope = silt::vec2 (
    grad.x * scale.z / scale.x,
    grad.y * scale.z / scale.y
  );

  return -(gravity / drag) * slope;

}

template<size_t K>                      //!< K = Transport Tensor Components
__global__ void __solve_uniform (
  silt::tensor_t<float> flux,           //!< Flux Integral Estimate [X*m^D/s]
  const silt::tensor_t<float> height,   //!< Height Field           [1]
  const silt::tensor_t<float> source,   //!< Source Term Tensor     [X/s]
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor      [1/s]
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source   [1]
  const silt::shape shape,              //!< Tensor Shape
  const silt::vec3 scale,               //!< Cell Scale             [m]
  const float gravity,                  //!< Gravity                [m/s^2]
  const float drag,                     //!< Drag Rate              [1/s]
  const float epsilon,                  //!< Minimum Attenuation    [1]
  const float maxstep                   //!< Maximum Particle Steps [1]
) {

  // Note: Number of concurrent samples given
  //  by the dimensionality of the rng tensor.
  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n >= rng.elem()) return;

  // Extract Correct Dimension Views
  constexpr size_t D = 2;                   //!< Dimensionality of Domain
  using vecD = silt::fvec<D>;               //!< Vector of D Components
  using vecK = silt::fvec<K>;               //!< Vector of K Components
  auto fluxView = flux.view<vecK>();        //!< Tensorised Flux View         [X/s]
  auto heightView = height.view<float>();   //!< Height Field View            [1]
  auto sourceView = source.view<vecK>();    //!< Tensorised Source View       [X/s]

  const silt::ivec2 res = silt::ivec2(shape[0], shape[1]);

  // Initialize Particle State
  float att = 1.0f;                         //!< Cumulative Flux Attenuation  [1]
  float lambda = 0.0f;                      //!< Cumulative Time Travelled    [s]
  vecD pos = vecD {                         //!< Position in Grid-Space       [1]
    curand_uniform(&rng[n])*float(shape[0]),
    curand_uniform(&rng[n])*float(shape[1])
  };
  int ind = shape.flatten(pos);             //!< Grid Tensor Index

  // Upstream Contributions Sample
  const silt::vec2 scaleXY = silt::vec2(scale.x, scale.y);
  const float L = glm::length(scaleXY);     //!< Length of Cell         [m]
  const float A = (scale.x * scale.y);      //!< Area of Cell           [m^D]
  const float P = 1.0f / (A * shape.elem());  //!< Probability of Sample  [1/m^D]
  const vecK S = sourceView[ind] / P;       //!< Sampled Source Rate    [X*m^D/s]
  if(glm::length(S) < epsilon)              //!< Early Termination
    return;

  // The particle carries its own momentum, so it is launched at the local
  // terminal velocity and relaxes from there as the gradient under it changes.
  vecD v = __velocity(heightView, res, pos, scale, gravity, drag);

  // Integrate along Streamline
  int step = 0;
  while(!shape.oob(pos) && epsilon < abs(att) && ++step < maxstep) {

    // Accumulate Estimate upon exiting Cell
    const int nind = shape.flatten(pos);
    if(nind != ind) {
      ind = nind;
      if constexpr(K >= 1){
        atomicAdd(&fluxView[ind].x, S.x * att);
      }
      if constexpr(K >= 2){
        atomicAdd(&fluxView[ind].y, S.y * att);
      }
      if constexpr(K >= 3){
        atomicAdd(&fluxView[ind].z, S.z * att);
      }
    }

    // The ordered comparison also rejects a NaN, which is what a gather just
    // outside its support returns: the last cell is in bounds for the position
    // test above but has no interpolation support.
    const float v_len = glm::length(v);         //!< Velocity [m/s]
    if(!(v_len > epsilon))
      break;

    // Update the Position in Grid-Space
    const silt::vec2 v_norm = v / v_len;        //!< Normalized Direction   [1]
    const float ds = __stepsize(pos, v_norm);   //!< Step-Size in Gridspace [1]
    pos += ds * v_norm;                         //!< Grid Position Update   [1]
    if(shape.oob(pos))
      break;

    // Update the Real Time and Attenuation. The step is the voxel-crossing
    // time at the current speed, which the quasi-static assumption licenses.
    const float dlambda = ds * L / v_len;       //!< Real Time Travelled  [t]
    att *= __expf(-dlambda * decay[ind]);       //!< Decay Attenuation    [1]
    lambda += dlambda;                          //!< Update Real Time     [t]

    // Relax the velocity onto the terminal velocity where the particle now is,
    // which is the exact solution for a gradient held fixed over the step. A
    // short step leaves the velocity nearly untouched, so the particle coasts
    // across an adverse gradient; a long one lands it on the local terminal
    // velocity, which is steepest descent.
    const vecD v_eq = __velocity(heightView, res, pos, scale, gravity, drag);
    v = v_eq + (v - v_eq) * __expf(-drag * dlambda);

  }

}

template<size_t K>                      //!< K = Transport Tensor Components
__global__ void __finalize (
  silt::tensor_t<float> flux,           //!< Flux Integral Estimate [X*m^D/s]
  const silt::tensor_t<float> source,   //!< Source Term            [X/s]
  const silt::vec2 scale,               //!< Cell Scale             [m]
  const size_t count                    //!< Sample Count           [1]
) {

  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n >= flux.elem()/K) return;

  using vecK = silt::fvec<K>;             //!< Vector of K Components
  auto fluxView = flux.view<vecK>();      //!< Tensorised Flux View   [X*m^D/s]
  auto sourceView = source.view<vecK>();  //!< Tensorised Source View [X/s]

  const float A = scale.x * scale.y;      //!< [m^D]

  // Note: We add the scaled source term here, so that even positions with zero samples
  //  have a positive value corresponding to their local source term. This additional
  //  value is NOT divided by the sample count.
  //
  // The estimate is left as the flux integral. Dividing by a cell-crossing rate
  // here would turn it into a concentration and fold an arbitrary constant into
  // whatever law consumes it.

  fluxView[n] = sourceView[n] * A + fluxView[n] / float(count);  //!< [X*m^D/s]

}

namespace soil {

//! Uniform Distribution Grid-Free Monte-Carlo
//!   Estimator for Linear Conservation Laws
//!
//! Note that this estimator is parameterized so that the 3rd dimension
//! of the source tensor is the dimensionality of the transported quantity.
//!
silt::tensor solve_uniform (
  const silt::tensor_t<float> height,   //!< Height Field           [1]
  const silt::tensor_t<float> source,   //!< Source Term Tensor     [X/s]
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor      [1/s]
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source   [1]
  const silt::vec3 scale,               //!< Cell Scale             [m]
  const size_t count,                   //!< Sample Count           [1]
  const float gravity,                  //!< Gravity                [m/s^2]
  const float drag                      //!< Drag Rate              [1/s]
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
  const silt::vec2 scaleXY = silt::vec2(scale.x, scale.y);

  // Resolve Data-Layout of Source / Flux Tensor
  switch(D) {
    case 1:
      __solve_uniform<1><<<block(rng.elem(), 512), 512>>>(flux, height, source, decay, rng, shape, scale, gravity, drag, epsilon, maxstep);
      gpuErrchk(cudaGetLastError());
      __finalize<1><<<block(flux.elem(), 512), 512>>>(flux, source, scaleXY, count);
      gpuErrchk(cudaGetLastError());
      break;
    case 2:
      __solve_uniform<2><<<block(rng.elem(), 512), 512>>>(flux, height, source, decay, rng, shape, scale, gravity, drag, epsilon, maxstep);
      gpuErrchk(cudaGetLastError());
      __finalize<2><<<block(flux.elem(), 512), 512>>>(flux, source, scaleXY, count);
      gpuErrchk(cudaGetLastError());
      break;
    case 3:
      __solve_uniform<3><<<block(rng.elem(), 512), 512>>>(flux, height, source, decay, rng, shape, scale, gravity, drag, epsilon, maxstep);
      gpuErrchk(cudaGetLastError());
      __finalize<3><<<block(flux.elem(), 512), 512>>>(flux, source, scaleXY, count);
      gpuErrchk(cudaGetLastError());
      break;
    default:
      break;
  }

  gpuErrchk(cudaDeviceSynchronize());
  return silt::tensor(flux);

}

}
