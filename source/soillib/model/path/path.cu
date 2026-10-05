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

//! Steepest descent at a position in grid-space, as a dimensionless slope.
__device__ silt::vec2 __descent (
  const silt::view_t<const float>& height, //!< Height Field          [1]
  const silt::ivec2 res,                   //!< Field Resolution      [1]
  const silt::vec2 pos,                    //!< Grid-Space Position   [1]
  const silt::vec3 scale                   //!< Cell Scale            [m]
) {

  // Per-cell differences become a dimensionless slope through scale.z/scale.xy.
  const auto support = sample_t<float, 2, 1>::gather(height, res, pos);
  const silt::vec2 grad = support.grad();

  return -silt::vec2 (
    grad.x * scale.z / scale.x,
    grad.y * scale.z / scale.y
  );

}

template<size_t K>                      //!< K = Transport Tensor Components
__global__ void __solve_uniform (
  silt::tensor_t<float> flux,           //!< Flux Integral Estimate [X*m^D/s]
  const silt::tensor_t<float> height,   //!< Height Field           [1]
  const silt::tensor_t<float> source,   //!< Source Term Tensor     [X/s]
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor      [1/m]
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source   [1]
  const silt::shape shape,              //!< Tensor Shape
  const silt::vec3 scale,               //!< Cell Scale             [m]
  const float drag,                     //!< Inverse Drag Length    [1/m]
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
  float arc = 0.0f;                         //!< Cumulative Path Length       [m]
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

  // From rest: one cell of relaxation, so the first direction is steepest descent.
  const vecD d_init = __descent(heightView, res, pos, scale);
  vecD d = d_init * (1.0f - __expf(-drag * L));

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

    // The ordered comparison also rejects the NaN a gather returns just outside
    // its support, where the position test above still passes.
    const float d_len = glm::length(d);         //!< Direction Magnitude  [1]
    if(!(d_len > epsilon))
      break;

    // Advance one cell along the direction.
    const silt::vec2 d_norm = d / d_len;        //!< Normalized Direction   [1]
    const float ds = __stepsize(pos, d_norm);   //!< Step-Size in Gridspace [1]
    pos += ds * d_norm;                         //!< Grid Position Update   [1]
    if(shape.oob(pos))
      break;

    const float darc = ds * L;                  //!< Path Length Travelled  [m]
    att *= __expf(-darc * decay[ind]);          //!< Decay Attenuation     [1]
    arc += darc;                                //!< Update Path Length    [m]

    // Relax the direction onto steepest descent over the step.
    const vecD d_eq = __descent(heightView, res, pos, scale);
    d = d_eq + (d - d_eq) * __expf(-drag * darc);

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

  // The local source term is added unscaled by the sample count, so a cell no
  // particle visited still carries it.
  fluxView[n] = sourceView[n] * A + fluxView[n] / float(count);  //!< [X*m^D/s]

}

namespace soil {

silt::tensor solve_uniform (
  const silt::tensor_t<float> height,   //!< Height Field           [1]
  const silt::tensor_t<float> source,   //!< Source Term Tensor     [X/time/m^D]
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor      [1/m]
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source   [1]
  const silt::vec3 scale,               //!< Cell Scale             [m]
  const size_t count,                   //!< Sample Count           [1]
  const float drag                      //!< Inverse Drag Length    [1/m]
) {

  // The flux carries the component count of the source term; the domain shape
  // drops the third dimension.
  const silt::shape shapeIn = source.shape();
  const size_t D = shapeIn[2];  // Dimensionality of Transported Quantity
  const silt::shape shape = silt::shape(shapeIn[0], shapeIn[1]);
  auto flux = silt::tensor_t<float>(shapeIn, silt::host_t::GPU);
  silt::set(flux, 0.0f);

  const float epsilon = 1E-16f;               //!< Minimum Attenuation [1]
  const float maxstep = shape[0] + shape[1];  //!< Manhattan Step Bound
  const silt::vec2 scaleXY = silt::vec2(scale.x, scale.y);

  switch(D) {
    case 1:
      __solve_uniform<1><<<block(rng.elem(), 512), 512>>>(flux, height, source, decay, rng, shape, scale, drag, epsilon, maxstep);
      gpuErrchk(cudaGetLastError());
      __finalize<1><<<block(flux.elem(), 512), 512>>>(flux, source, scaleXY, count);
      gpuErrchk(cudaGetLastError());
      break;
    case 2:
      __solve_uniform<2><<<block(rng.elem(), 512), 512>>>(flux, height, source, decay, rng, shape, scale, drag, epsilon, maxstep);
      gpuErrchk(cudaGetLastError());
      __finalize<2><<<block(flux.elem(), 512), 512>>>(flux, source, scaleXY, count);
      gpuErrchk(cudaGetLastError());
      break;
    case 3:
      __solve_uniform<3><<<block(rng.elem(), 512), 512>>>(flux, height, source, decay, rng, shape, scale, drag, epsilon, maxstep);
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
