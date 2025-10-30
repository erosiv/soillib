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

//
// Laplacian Implementation
//

template<size_t D>
__global__ void __laplacian (
  silt::tensor_t<float> tensorOut,      //!< Output Field
  const silt::tensor_t<float> tensorIn, //!< Input Field
  const silt::shape shape,              //!< Laplacian Field Shape
  const silt::vec2 scale
) {

  using vec = silt::fvec<D>;

  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
  if(n >= shape.elem) return;

  auto viewIn = tensorIn.view<vec>();   // In Vector View
  auto viewOut = tensorOut.view<vec>(); // Out Vector View

  // Boundary Continuation Laplacian:
  //  Note that this can introduce strong diffusion at boundary.
  const silt::ivec2 ipos = shape.unflatten(n);
  const vec v00 = viewIn[n];
  const vec vn0 = shape.oob(ipos + silt::ivec2(-1, 0)) ? v00 : viewIn[shape.flatten(ipos + silt::ivec2(-1, 0))];
  const vec vp0 = shape.oob(ipos + silt::ivec2( 1, 0)) ? v00 : viewIn[shape.flatten(ipos + silt::ivec2( 1, 0))];
  const vec v0n = shape.oob(ipos + silt::ivec2( 0,-1)) ? v00 : viewIn[shape.flatten(ipos + silt::ivec2( 0,-1))];
  const vec v0p = shape.oob(ipos + silt::ivec2( 0, 1)) ? v00 : viewIn[shape.flatten(ipos + silt::ivec2( 0, 1))];
  vec lx = (vn0 + vp0 - 2.0f * v00)/scale.x/scale.x;
  vec ly = (v0n + v0p - 2.0f * v00)/scale.y/scale.y;

  viewOut[n] = lx + ly;

}

//! 2D Tensor Laplacian
silt::tensor_t<float> laplacian(const silt::tensor_t<float>& tensor, const silt::vec2 scale) {

  // basically implement the component-wise laplacian function...
  //  should return the exact same dimsensionality of the previous vector.

  const silt::shape shapeIn = tensor.shape();
  const silt::shape shape = silt::shape(shapeIn[0], shapeIn[1]);

  auto laplacian = silt::tensor_t<float>(shapeIn, silt::host_t::GPU);

  if(shapeIn[2] == 1) {
    __laplacian<1><<<block(shape.elem, 512), 512>>>(laplacian, tensor, shape, scale);
  }

  if(shapeIn[2] == 2) {
    __laplacian<2><<<block(shape.elem, 512), 512>>>(laplacian, tensor, shape, scale);
  }

  return laplacian;

}

} // end of namespace soil

#endif