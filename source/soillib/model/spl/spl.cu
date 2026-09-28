#include <soillib/soillib.hpp>

#include <soillib/model/spl/spl.hpp>

namespace soil {

namespace {

inline int block(const int64_t elem, const int thread) {
  return int((elem + thread - 1) / thread);
}

}

//
// Stream-Power Rate Kernel
//

__global__ void __spl_rate (
  silt::tensor_t<float> rate,             //!< Output Erosion Rate
  const silt::tensor_t<float> discharge,  //!< Input Discharge Field
  const silt::tensor_t<float> slope,      //!< Input Slope Field
  const float K,
  const float m,
  const float n,
  const silt::shape shape
){

  const int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(i >= shape.elem())
    return;

  const float Q = fmaxf(discharge[i], 0.0f);
  const float S = fabsf(slope[i]);
  rate[i] = K * __powf(Q, m) * __powf(S, n);

}

void spl_rate (
  silt::tensor_t<float> rate,
  const silt::tensor_t<float> discharge,
  const silt::tensor_t<float> slope,
  const float K,
  const float m,
  const float n
) {

  if(discharge.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, discharge.host());

  const silt::shape shape = discharge.shape();
  __spl_rate<<<block(shape.elem(), 512), 512>>>(rate, discharge, slope, K, m, n, shape);
  gpuErrchk(cudaGetLastError());

}

//
// Height Update Kernel
//

__global__ void __spl_apply (
  silt::tensor_t<float> height,      //!< Height Field, Updated in Place
  const silt::tensor_t<float> rate,  //!< Input Erosion Rate
  const float dt,
  const silt::shape shape
){

  const int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(i >= shape.elem())
    return;

  height[i] -= dt * rate[i];

}

void spl_apply (
  silt::tensor_t<float> height,
  const silt::tensor_t<float> rate,
  const float dt
) {

  if(height.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, height.host());

  const silt::shape shape = height.shape();
  __spl_apply<<<block(shape.elem(), 512), 512>>>(height, rate, dt, shape);
  gpuErrchk(cudaGetLastError());

}

//
// Stream-Power Diffusivity Kernel
//

__global__ void __spl_diffusivity (
  silt::tensor_t<float> diffusivity,      //!< Output Diffusivity Field
  const silt::tensor_t<float> discharge,  //!< Input Discharge Field
  const silt::tensor_t<float> slope,      //!< Input Slope Field
  const float D0,
  const float k_D,
  const float m,
  const float n,
  const silt::shape shape
){

  const int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(i >= shape.elem())
    return;

  const float Q = fmaxf(discharge[i], 0.0f);
  const float S = fabsf(slope[i]);
  diffusivity[i] = D0 + fmaxf(k_D * powf(Q, m) * powf(S + 1E-8f, n), 0.0f);

}

void spl_diffusivity (
  silt::tensor_t<float> diffusivity,
  const silt::tensor_t<float> discharge,
  const silt::tensor_t<float> slope,
  const float D0,
  const float k_D,
  const float m,
  const float n
) {

  if(discharge.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, discharge.host());

  const silt::shape shape = discharge.shape();
  __spl_diffusivity<<<block(shape.elem(), 512), 512>>>(diffusivity, discharge, slope, D0, k_D, m, n, shape);
  gpuErrchk(cudaGetLastError());

}

//
// Deposition Rate Kernel
//

__global__ void __spl_deposition (
  silt::tensor_t<float> deposition,  //!< Output Deposition Rate
  const silt::tensor_t<float> Qs,    //!< Input Sediment Discharge
  const silt::tensor_t<float> Qw,    //!< Input Water Discharge
  const float k_S,
  const silt::shape shape
){

  const int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(i >= shape.elem())
    return;

  const float eps = 1e-6f;
  deposition[i] = k_S * Qs[i] / fmaxf(Qw[i], eps);

}

void spl_deposition (
  silt::tensor_t<float> deposition,
  const silt::tensor_t<float> Qs,
  const silt::tensor_t<float> Qw,
  const float k_S
) {

  if(Qs.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, Qs.host());

  const silt::shape shape = Qs.shape();
  __spl_deposition<<<block(shape.elem(), 512), 512>>>(deposition, Qs, Qw, k_S, shape);
  gpuErrchk(cudaGetLastError());

}

} // end of namespace soil
