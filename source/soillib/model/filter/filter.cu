#include <soillib/soillib.hpp>

#include <soillib/model/filter/filter.hpp>
#include <math_constants.h>
#include <soillib/model/graph/graph.hpp>

namespace soil {

namespace {

inline int block(const int64_t elem, const int thread) {
  return int((elem + thread - 1) / thread);
}

}

//
// Gaussian Blur Kernel
//

template<typename T>
__device__ void __gaussian_blur (
  silt::view_t<T>& bufferOut,
  const silt::view_t<T>& bufferIn,
  const silt::shape& shape,
  const unsigned int n,
  const float sigma,
  const bool xdir
){
   
  const silt::ivec2 pos = shape.unflatten(n);       //!< 2D Position
  const float s = fmaxf(sigma, 1e-6f);
  const float k2 = 0.5f / (s * s);
  const int kwindow = int(fminf(ceilf(3.0f * s), 16.0f));

  T val = T{0};                               //!< Blurred Value
  float Z = 0.0f;                             //!< Discrete Weight Sum

  for(int k = -kwindow; k <= kwindow; ++k){

    silt::ivec2 npos = pos + (xdir ? silt::ivec2(k, 0) : silt::ivec2(0, k));
    if(npos.x < 0) npos.x = 0;
    if(npos.y < 0) npos.y = 0;
    if(npos.x > shape[0] - 1) npos.x = shape[0] - 1;
    if(npos.y > shape[1] - 1) npos.y = shape[1] - 1;
    const int nind = shape.flatten(npos);

    const float w = __expf(-k2 * float(k) * float(k));
    val += bufferIn[nind] * w;
    Z += w;

  }

  bufferOut[n] = val * (1.0f / Z);

}

template<typename T>
__global__ void __blur (
  silt::view_t<T> tensorOut,      //!< Output Field
  const silt::view_t<T> tensorIn, //!< Input Field
  const silt::shape shape,
  const float sigma,
  const bool xdir
){
  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n < shape.elem()) {
    __gaussian_blur<T>(tensorOut, tensorIn, shape, n, sigma, xdir);
  };
}

silt::tensor_t<float> gaussian_blur(silt::tensor_t<float> tensorIn, const float sigma) {

  const silt::shape shapeIn = tensorIn.shape();
  const silt::shape shape(shapeIn[0], shapeIn[1]);
  const int channel = shapeIn[2];
  silt::tensor_t<float> tensorOut(shapeIn, silt::host_t::GPU);

  if(channel == 1) {
    using T = silt::vec<1>;
    __blur<T><<<block(shape.elem(), 512), 512>>>(tensorOut.view<T>(), tensorIn.view<T>(), shape, sigma, true);
    gpuErrchk(cudaGetLastError());
    __blur<T><<<block(shape.elem(), 512), 512>>>(tensorIn.view<T>(), tensorOut.view<T>(), shape, sigma, false);
    gpuErrchk(cudaGetLastError());
  }
  if(channel == 2) {
    using T = silt::vec<2>;
    __blur<T><<<block(shape.elem(), 512), 512>>>(tensorOut.view<T>(), tensorIn.view<T>(), shape, sigma, true);
    gpuErrchk(cudaGetLastError());
    __blur<T><<<block(shape.elem(), 512), 512>>>(tensorIn.view<T>(), tensorOut.view<T>(), shape, sigma, false);
    gpuErrchk(cudaGetLastError());
  }
  if(channel == 3) {
    using T = silt::vec<3>;
    __blur<T><<<block(shape.elem(), 512), 512>>>(tensorOut.view<T>(), tensorIn.view<T>(), shape, sigma, true);
    gpuErrchk(cudaGetLastError());
    __blur<T><<<block(shape.elem(), 512), 512>>>(tensorIn.view<T>(), tensorOut.view<T>(), shape, sigma, false);
    gpuErrchk(cudaGetLastError());
  }

  return tensorIn;
}

//
// Gaussian Blur Kernel, Spatially-Varying Sigma
//
//  Still separable, still a fixed kwindow -- sigma is just looked up per
//  output cell instead of passed as one constant. An approximation (a true
//  space-varying Gaussian isn't separable), but a cheap and standard one.
//

template<typename T>
__global__ void __blur_varying (
  silt::view_t<T> tensorOut,           //!< Output Field
  const silt::view_t<T> tensorIn,      //!< Input Field
  const silt::tensor_t<float> sigma,   //!< Per-Cell Sigma Field
  const silt::shape shape,
  const bool xdir
){
  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n < shape.elem()) {
    __gaussian_blur<T>(tensorOut, tensorIn, shape, n, sigma[n], xdir);
  };
}

silt::tensor_t<float> gaussian_blur(silt::tensor_t<float> tensorIn, const silt::tensor_t<float> sigma) {

  if(sigma.host() != silt::host_t::GPU)
    throw silt::error::mismatch_host(silt::host_t::GPU, sigma.host());

  const silt::shape shapeIn = tensorIn.shape();
  const silt::shape shape(shapeIn[0], shapeIn[1]);
  const int channel = shapeIn[2];
  silt::tensor_t<float> tensorOut(shapeIn, silt::host_t::GPU);

  if(channel == 1) {
    using T = silt::vec<1>;
    __blur_varying<T><<<block(shape.elem(), 512), 512>>>(tensorOut.view<T>(), tensorIn.view<T>(), sigma, shape, true);
    gpuErrchk(cudaGetLastError());
    __blur_varying<T><<<block(shape.elem(), 512), 512>>>(tensorIn.view<T>(), tensorOut.view<T>(), sigma, shape, false);
    gpuErrchk(cudaGetLastError());
  }
  if(channel == 2) {
    using T = silt::vec<2>;
    __blur_varying<T><<<block(shape.elem(), 512), 512>>>(tensorOut.view<T>(), tensorIn.view<T>(), sigma, shape, true);
    gpuErrchk(cudaGetLastError());
    __blur_varying<T><<<block(shape.elem(), 512), 512>>>(tensorIn.view<T>(), tensorOut.view<T>(), sigma, shape, false);
    gpuErrchk(cudaGetLastError());
  }
  if(channel == 3) {
    using T = silt::vec<3>;
    __blur_varying<T><<<block(shape.elem(), 512), 512>>>(tensorOut.view<T>(), tensorIn.view<T>(), sigma, shape, true);
    gpuErrchk(cudaGetLastError());
    __blur_varying<T><<<block(shape.elem(), 512), 512>>>(tensorIn.view<T>(), tensorOut.view<T>(), sigma, shape, false);
    gpuErrchk(cudaGetLastError());
  }

  return tensorIn;
}

} // end of namespace soil
