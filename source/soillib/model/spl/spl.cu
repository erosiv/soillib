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
// Exponential Fitting: Relaxation Rate + Fitted Update
//

__global__ void __spl_decay (
  silt::tensor_t<float> decay,            //!< Output Decay Rate    [1/m]
  const silt::tensor_t<float> discharge,  //!< Input Discharge      [m^3/time]
  const float k_S,
  const float width,
  const float minimum,
  const silt::shape shape
){

  const int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(i >= shape.elem())
    return;

  decay[i] = k_S * width / fmaxf(discharge[i], minimum);

}

void spl_decay (
  silt::tensor_t<float> decay,
  const silt::tensor_t<float> discharge,
  const float k_S,
  const float width,
  const float minimum
) {

  if(discharge.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, discharge.host());

  const silt::shape shape = discharge.shape();
  __spl_decay<<<block(shape.elem(), 512), 512>>>(decay, discharge, k_S, width, minimum, shape);
  gpuErrchk(cudaGetLastError());

}

__global__ void __spl_relax (
  silt::tensor_t<float> relax,            //!< Output Relaxation Rate
  const silt::tensor_t<float> discharge,  //!< Input Discharge Field
  const silt::tensor_t<float> slope,      //!< Input Slope Field
  const float K,
  const float m,
  const float n,
  const float dSdh,                       //!< Slope Sensitivity, 1/dx w. z-fold
  const silt::shape shape
){

  const int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(i >= shape.elem())
    return;

  const float Q = fmaxf(discharge[i], 0.0f);
  const float S = fmaxf(fabsf(slope[i]), 1e-8f);
  relax[i] = fmaxf(n * K * powf(Q, m) * powf(S, n - 1.0f) * dSdh, 0.0f);

}

void spl_relax (
  silt::tensor_t<float> relax,
  const silt::tensor_t<float> discharge,
  const silt::tensor_t<float> slope,
  const float K,
  const float m,
  const float n,
  const silt::vec3 scale
) {

  if(discharge.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, discharge.host());

  // Steeper axis => larger lambda => more damping.
  const float dSdh = scale.z / fminf(scale.x, scale.y);

  const silt::shape shape = discharge.shape();
  __spl_relax<<<block(shape.elem(), 512), 512>>>(relax, discharge, slope, K, m, n, dSdh, shape);
  gpuErrchk(cudaGetLastError());

}

__global__ void __spl_apply_exp (
  silt::tensor_t<float> height,      //!< Height Field, Updated in Place
  const silt::tensor_t<float> rate,  //!< Input Net Rate
  const silt::tensor_t<float> relax, //!< Input Relaxation Rate
  const float dt,
  const silt::shape shape
){

  const int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(i >= shape.elem())
    return;

  const float z = fmaxf(relax[i] * dt, 0.0f);

  // phi(z) = (1 - exp(-z))/z. Series below ~1e-4 avoids 0/0 and the
  // cancellation in (1 - exp(-z)) for small z.
  const float phi = (z > 1e-4f) ? (-expm1f(-z) / z)
                                : (1.0f - 0.5f * z + z * z / 6.0f);

  height[i] -= dt * phi * rate[i];

}

void spl_apply_exp (
  silt::tensor_t<float> height,
  const silt::tensor_t<float> rate,
  const silt::tensor_t<float> relax,
  const float dt
) {

  if(height.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, height.host());

  const silt::shape shape = height.shape();
  __spl_apply_exp<<<block(shape.elem(), 512), 512>>>(height, rate, relax, dt, shape);
  gpuErrchk(cudaGetLastError());

}

//
// Deposition Spill Limiter
//

__global__ void __spill_limit (
  silt::tensor_t<float> heightOut,        //!< Output Clamped Height
  const silt::tensor_t<float> height,     //!< Input Updated Height
  const silt::tensor_t<float> before,     //!< Input Pre-Step Height
  const float epsilon,
  const silt::shape shape
){

  const int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(i >= shape.elem())
    return;

  const silt::ivec2 ipos = shape.unflatten(i);
  const float h0 = before[i];

  const int sx[4] = {-1, 1,  0, 0};
  const int sy[4] = { 0, 0, -1, 1};

  // Donors: neighbors that stood above this cell before the step. The cell
  // may rise to their updated level at most.
  float spill = 0.0f;
  bool bounded = false;

  for(int k = 0; k < 4; ++k){
    const silt::ivec2 npos = silt::ivec2(ipos.x + sx[k], ipos.y + sy[k]);
    if(shape.oob(npos))
      continue;
    const int64_t j = shape.flatten(npos);
    if(before[j] <= h0)
      continue;
    const float hj = height[j];
    spill = bounded ? fminf(spill, hj) : hj;
    bounded = true;
  }

  const float h = height[i];
  heightOut[i] = bounded ? fminf(h, spill - epsilon) : h;

}

silt::tensor_t<float> spill_limit (
  const silt::tensor_t<float>& height,
  const silt::tensor_t<float>& before,
  const float epsilon
) {

  if(height.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, height.host());

  const silt::shape shape = height.shape();
  auto out = silt::tensor_t<float>(shape, silt::host_t::GPU);
  __spill_limit<<<block(shape.elem(), 512), 512>>>(out, height, before, epsilon, shape);
  gpuErrchk(cudaGetLastError());
  return out;

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
