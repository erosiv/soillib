#ifndef SOILLIB_MODEL_FILTER_CU
#define SOILLIB_MODEL_FILTER_CU
#define HAS_CUDA

#include <soillib/model/filter/filter.hpp>

namespace soil {

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

}

//
// Gaussian Blur Kernel
//

template<typename T>
__device__ void __gaussian_blur (
  silt::tensor_t<T>& bufferOut,
  const silt::tensor_t<T>& bufferIn,
  const silt::shape& shape,
  const unsigned int n,
  const float sigma,
  const bool xdir
){
   
  const silt::ivec2 pos = shape.unflatten(n);       //!< 2D Position
  const int kwindow = 16;                     //!< Kernel Window Size
  T val = T{0};                               //!< Blurred Value

  for(int k = -kwindow; k <= kwindow; ++k){

    silt::ivec2 npos = pos + (xdir ? silt::ivec2(k, 0) : silt::ivec2(0, k));
    if(npos.x < 0) npos.x = 0;
    if(npos.y < 0) npos.y = 0;
    if(npos.x > shape[0] - 1) npos.x = shape[0] - 1;
    if(npos.y > shape[1] - 1) npos.y = shape[1] - 1;
    const int nind = shape.flatten(npos);

//    const float sigma2 = 
    const float Z = sqrt(2.0f * 3.14159265f) * sigma;
    const float kernel = __expf(-0.5f*(k / sigma)*(k / sigma)) / Z;
    const float src = bufferIn[nind];
    val += src * kernel;

  }

  bufferOut[n] = val;

}

__global__ void __blur(
  silt::tensor_t<float> tensorOut,      //!< Output Field
  const silt::tensor_t<float> tensorIn, //!< Input Field
  const silt::shape shape,
  const float sigma,
  const bool xdir
){
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n < shape.elem){
    __gaussian_blur<float>(tensorOut, tensorIn, shape, n, sigma, xdir);
  };
}

silt::tensor_t<float> gaussian_blur(silt::tensor_t<float> tensorIn, const float sigma){
  const silt::shape shape = tensorIn.shape();
  silt::tensor_t<float> tensorOut(shape, silt::host_t::GPU);
  __blur<<<block(shape.elem, 512), 512>>>(tensorOut, tensorIn, shape, sigma, true);
  __blur<<<block(shape.elem, 512), 512>>>(tensorIn, tensorOut, shape, sigma, false);
  return tensorIn;
}

} // end of namespace soil

#endif