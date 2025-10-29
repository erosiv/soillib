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

__global__ void __solve_uniform (
  silt::tensor_t<float> flux,           //!< Flux Integral Estimate []
  const silt::tensor_t<float> flow,     //!< Flow-Field Tensor      []
  const silt::tensor_t<float> source,   //!< Source Term Tensor     [X/s]
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor      [1/s]
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source
  const silt::shape shape,              //!< Tensor Shape
  const silt::vec2 scale,               //!< Cell Scale
  const float lambda_max,               //!< Maximum Char. Time
  const float epsilon                   //!< Minimum Attenuation
) {

  // Note: Number of concurrent samples given
  //  by the dimensionality of the rng tensor.
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= rng.elem()) return;

  // Constant Things...
  const auto view = flow.view<silt::vec2>();

  // Initialize Particle
  float att = 1.0f;
  float lambda = 0.0f;
  silt::vec2 pos = silt::vec2 {
    curand_uniform(&rng[n])*float(shape[0]),
    curand_uniform(&rng[n])*float(shape[1])
  };
  int ind = shape.flatten(pos);

  // Upstream Contributions Sample
  const float P = 1.0f / float(shape.elem);
  const float S = source[ind] / P;

  // Integrate along Streamline
  int step = 0;
  while(!shape.oob(pos) && (lambda < lambda_max) && (abs(att) > epsilon) && ++step < 1024) {

    // Accumulate Estimate
    const int nind = shape.flatten(pos);  // New Index?
    if(nind != ind) {
      ind = nind;
      atomicAdd(&flux[ind], S * att);
    }

    const sample_t<silt::vec2, 2, 1> v_support = sample_t<silt::vec2, 2, 1>::gather(view, silt::ivec2(shape[0], shape[1]), pos);
    const silt::vec2 v = v_support.val();
    if(glm::length(v) < 1E-8)
      break;

    const float dlambda = glm::length(scale / v);
    att *= __expf(-dlambda * decay[ind]);
    pos += 1.414f * v / glm::length(v);
    lambda += dlambda;

  }

}

__global__ void __normalize (
  silt::tensor_t<float> flux,           //!< Flux Integral Estimate []
  const silt::tensor_t<float> flow,     //!< Flow-Field Tensor      []
  const silt::vec2 scale,               //!< Cell Scale
  const size_t count                    //!< Sample Count
) {

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= flux.elem()) return;

  const auto view = flow.view<silt::vec2>();
  const silt::vec2 v = view[n];
  flux[n] = flux[n] / float(count);// / abs(glm::dot(v, scale));

}

namespace soil {

silt::tensor solve_uniform (
  const silt::tensor_t<float> flow,     //!< Flow-Field Tensor      []
  const silt::tensor_t<float> source,   //!< Source Term Tensor     [X/s]
  const silt::tensor_t<float> decay,    //!< Decay Term Tensor      [1/s]
  silt::tensor_t<silt::rng> rng,        //!< Random Number Source
  const silt::vec2 scale,               //!< Cell Scale
  const size_t count                    //!< Sample Count
) {

  // 
  // Uniform Solution Algorithm:
  //

  const float epsilon = 1e-3;
  const float lambda_max = 512;

  const silt::shape shape = source.shape();
  auto flux = silt::tensor_t<float>(shape, silt::host_t::GPU);

  silt::set(flux, 0.0f);
  cudaDeviceSynchronize();

  // Split Sampling
//  const size_t K = block(count, rng.elem());
//  for(size_t k = 0; k <= K; ++k) {
//    const size_t n = (rng.elem() * k <= count) ? rng.elem() : (count % (rng.elem() * k));
//  }
  __solve_uniform<<<block(rng.elem(), 512), 512>>>(flux, flow, source, decay, rng, shape, scale, lambda_max, epsilon);
  __normalize<<<block(flux.elem(), 512), 512>>>(flux, flow, scale, count);
  cudaDeviceSynchronize();

  return silt::tensor(flux);

}

}

#endif