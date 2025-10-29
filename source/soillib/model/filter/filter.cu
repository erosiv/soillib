#ifndef SOILLIB_MODEL_FILTER_CU
#define SOILLIB_MODEL_FILTER_CU
#define HAS_CUDA

#include <soillib/model/filter/filter.hpp>
#include <math_constants.h>

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

//
// Gradient Implementation
//

__global__ void __gradient (
  silt::tensor_t<float> tensorOut,      //!< Output Field
  const silt::tensor_t<float> tensorIn, //!< Input Field
  const silt::shape shape,              //!< Input Field Shape
  const silt::vec2 scale
){

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem) return;
  
  // Data Loading w. Bounds Handling  
  const silt::ivec2 ipos = shape.unflatten(n);
  const float h = tensorIn[n];
  const float hn0 = shape.oob(ipos + silt::ivec2(-1, 0)) ? CUDART_NAN_F : tensorIn[shape.flatten(ipos + silt::ivec2(-1, 0))];
  const float hp0 = shape.oob(ipos + silt::ivec2( 1, 0)) ? CUDART_NAN_F : tensorIn[shape.flatten(ipos + silt::ivec2( 1, 0))];
  const float h0n = shape.oob(ipos + silt::ivec2( 0,-1)) ? CUDART_NAN_F : tensorIn[shape.flatten(ipos + silt::ivec2( 0,-1))];
  const float h0p = shape.oob(ipos + silt::ivec2( 0, 1)) ? CUDART_NAN_F : tensorIn[shape.flatten(ipos + silt::ivec2( 0, 1))];

  // Min Gradient Computation w. Bounds Handling
  float gx = 0.0f;
  if(__isnanf(hn0)) gx = (hp0 - h)/scale.x;
  if(__isnanf(hp0)) gx = (h - hn0)/scale.x;
  if(!__isnanf(hp0) && !__isnanf(hn0)){
    if(hn0 < hp0) gx = (h - hn0)/scale.x;
    if(hp0 < hn0) gx = (hp0 - h)/scale.x;
    // if they are the same, slope is zero
  }

  float gy = 0.0f;
  if(__isnanf(h0n)) gy = (h0p - h)/scale.y;
  if(__isnanf(h0p)) gy = (h - h0n)/scale.y;
  if(!__isnanf(h0p) && !__isnanf(h0n)){
    if(h0n < h0p) gy = (h - h0n)/scale.y;
    if(h0p < h0n) gy = (h0p - h)/scale.y;
    // if they are the same, slope is zero
  }

  // Write to 2D vector view
  auto view = tensorOut.view<silt::vec2>();
  view[n] = silt::vec2(gx, gy);

}

silt::tensor_t<float> gradient(const silt::tensor_t<float>& tensor, const silt::vec2 scale){

  const silt::shape shapeIn = tensor.shape();
  const silt::shape shapeOut = silt::shape(shapeIn[0], shapeIn[1], 2);
  auto gradient = silt::tensor_t<float>(shapeOut, silt::host_t::GPU);
  __gradient<<<block(shapeIn.elem, 512), 512>>>(gradient, tensor, shapeIn, scale);
  return gradient;

}

} // end of namespace soil

#endif