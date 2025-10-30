#ifndef SOILLIB_MODEL_GRAD_CU
#define SOILLIB_MODEL_GRAD_CU
#define HAS_CUDA

#include <soillib/model/grad/grad.hpp>
#include <math_constants.h>

namespace soil {

namespace {

inline int block(const int elem, const int thread) {
  return (elem + thread - 1) / thread;
}

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