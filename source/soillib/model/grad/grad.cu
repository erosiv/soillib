#include <soillib/soillib.hpp>

#include <soillib/model/grad/grad.hpp>
#include <soillib/model/graph/graph.hpp>
#include <math_constants.h>

namespace soil {

namespace {

inline int block(const int64_t elem, const int thread) {
  return int((elem + thread - 1) / thread);
}

}

//
// Gradient, Linear Descending (Steepest) Upwind
//

__global__ void __grad_steepest (
  silt::tensor_t<float> tensorOut,      //!< Output Gradient Field
  const silt::tensor_t<float> tensorIn, //!< Input Height Field
  const silt::shape shape,
  const silt::vec2 scale
){

  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n >= shape.elem()) return;

  const silt::ivec2 ipos = shape.unflatten(n);
  const D4_t dir;
  const float hlocal = tensorIn[n];

  // Steepest-neighbor selection by raw height drop (grid-unit distance,
  // matching the flow-graph's own steepest() / direction() selection).
  float dmax = 0.0f;
  silt::ivec2 bestShift(0, 0);
  bool found = false;

  for(int k = 0; k < D4_t::K; ++k){
    const silt::ivec2 shift = dir.shift[k];
    const silt::ivec2 npos = ipos + shift;
    if(shape.oob(npos))
      continue;
    const float drop = hlocal - tensorIn[shape.flatten(npos)];
    if(drop > dmax){
      dmax = drop;
      bestShift = shift;
      found = true;
    }
  }

  // Only the axis of the chosen neighbor gets a nonzero component -- the
  // one-sided difference toward it, in standard dz/dx sign convention.
  float gx = 0.0f;
  float gy = 0.0f;
  if(found){
    const float hn = tensorIn[shape.flatten(ipos + bestShift)];
    if(bestShift.x != 0) gx = (hn - hlocal) / (float(bestShift.x) * scale.x);
    if(bestShift.y != 0) gy = (hn - hlocal) / (float(bestShift.y) * scale.y);
  }

  auto view = tensorOut.view<silt::vec2>();
  view[n] = silt::vec2(gx, gy);

}

silt::tensor_t<float> grad_steepest(const silt::tensor_t<float>& tensor, const silt::vec3 scale){

  const silt::shape shapeIn = tensor.shape();
  const silt::shape shapeOut = silt::shape(shapeIn[0], shapeIn[1], 2);
  const silt::vec2 gscale(scale.x / scale.z, scale.y / scale.z);
  auto grad = silt::tensor_t<float>(shapeOut, silt::host_t::GPU);
  __grad_steepest<<<block(shapeIn.elem(), 512), 512>>>(grad, tensor, shapeIn, gscale);
  gpuErrchk(cudaGetLastError());
  return grad;

}

//
// Gradient, Quadratic Descending Upwind (Godunov/eikonal upwind)
//

__global__ void __grad_quadratic (
  silt::tensor_t<float> tensorOut,      //!< Output Gradient Field
  const silt::tensor_t<float> tensorIn, //!< Input Height Field
  const silt::shape shape,
  const silt::vec2 scale
){

  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n >= shape.elem()) return;

  const silt::ivec2 ipos = shape.unflatten(n);
  const float h = tensorIn[n];
  const float hn0 = shape.oob(ipos + silt::ivec2(-1, 0)) ? CUDART_NAN_F : tensorIn[shape.flatten(ipos + silt::ivec2(-1, 0))];
  const float hp0 = shape.oob(ipos + silt::ivec2( 1, 0)) ? CUDART_NAN_F : tensorIn[shape.flatten(ipos + silt::ivec2( 1, 0))];
  const float h0n = shape.oob(ipos + silt::ivec2( 0,-1)) ? CUDART_NAN_F : tensorIn[shape.flatten(ipos + silt::ivec2( 0,-1))];
  const float h0p = shape.oob(ipos + silt::ivec2( 0, 1)) ? CUDART_NAN_F : tensorIn[shape.flatten(ipos + silt::ivec2( 0, 1))];

  // Per axis: the steeper of the two one-sided downhill differences, zero
  // if both neighbors are uphill -- the eikonal upwind |grad h|, signed as
  // dz/dx. Descent toward -x is dz/dx > 0, toward +x is dz/dx < 0. An
  // out-of-bounds descent becomes 0 and so never passes the > 0 test.
  const float dxn = __isnanf(hn0) ? 0.0f : (h - hn0);  // descent toward -x
  const float dxp = __isnanf(hp0) ? 0.0f : (h - hp0);  // descent toward +x

  float gx = 0.0f;
  if(dxn > 0.0f || dxp > 0.0f)
    gx = (dxn >= dxp) ? (dxn / scale.x) : (-dxp / scale.x);

  const float dyn = __isnanf(h0n) ? 0.0f : (h - h0n);  // descent toward -y
  const float dyp = __isnanf(h0p) ? 0.0f : (h - h0p);  // descent toward +y

  float gy = 0.0f;
  if(dyn > 0.0f || dyp > 0.0f)
    gy = (dyn >= dyp) ? (dyn / scale.y) : (-dyp / scale.y);

  auto view = tensorOut.view<silt::vec2>();
  view[n] = silt::vec2(gx, gy);

}

silt::tensor_t<float> grad_quadratic(const silt::tensor_t<float>& tensor, const silt::vec3 scale){

  const silt::shape shapeIn = tensor.shape();
  const silt::shape shapeOut = silt::shape(shapeIn[0], shapeIn[1], 2);
  const silt::vec2 gscale(scale.x / scale.z, scale.y / scale.z);
  auto grad = silt::tensor_t<float>(shapeOut, silt::host_t::GPU);
  __grad_quadratic<<<block(shapeIn.elem(), 512), 512>>>(grad, tensor, shapeIn, gscale);
  gpuErrchk(cudaGetLastError());
  return grad;

}

//
// Gradient, Centered Finite Differences (5-point stencil)
//

__global__ void __grad_centered (
  silt::tensor_t<float> tensorOut,      //!< Output Gradient Field
  const silt::tensor_t<float> tensorIn, //!< Input Height Field
  const silt::shape shape,
  const silt::vec2 scale
){

  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n >= shape.elem()) return;

  const silt::ivec2 ipos = shape.unflatten(n);
  const float h = tensorIn[n];
  const float hn0 = shape.oob(ipos + silt::ivec2(-1, 0)) ? CUDART_NAN_F : tensorIn[shape.flatten(ipos + silt::ivec2(-1, 0))];
  const float hp0 = shape.oob(ipos + silt::ivec2( 1, 0)) ? CUDART_NAN_F : tensorIn[shape.flatten(ipos + silt::ivec2( 1, 0))];
  const float h0n = shape.oob(ipos + silt::ivec2( 0,-1)) ? CUDART_NAN_F : tensorIn[shape.flatten(ipos + silt::ivec2( 0,-1))];
  const float h0p = shape.oob(ipos + silt::ivec2( 0, 1)) ? CUDART_NAN_F : tensorIn[shape.flatten(ipos + silt::ivec2( 0, 1))];

  // Centered difference, falling back to a one-sided difference at the
  // boundary, and to 0 if that's unavailable too (e.g. a 1-cell axis).
  const float gxn = (h - hn0) / scale.x;
  const float gxp = (hp0 - h) / scale.x;
  const float gyn = (h - h0n) / scale.y;
  const float gyp = (h0p - h) / scale.y;

  float gx = 0.5f * (hp0 - hn0) / scale.x;
  if(__isnanf(gx)) gx = gxn;
  if(__isnanf(gx)) gx = gxp;
  if(__isnanf(gx)) gx = 0.0f;

  float gy = 0.5f * (h0p - h0n) / scale.y;
  if(__isnanf(gy)) gy = gyn;
  if(__isnanf(gy)) gy = gyp;
  if(__isnanf(gy)) gy = 0.0f;

  auto view = tensorOut.view<silt::vec2>();
  view[n] = silt::vec2(gx, gy);

}

silt::tensor_t<float> grad_centered(const silt::tensor_t<float>& tensor, const silt::vec3 scale){

  const silt::shape shapeIn = tensor.shape();
  const silt::shape shapeOut = silt::shape(shapeIn[0], shapeIn[1], 2);
  const silt::vec2 gscale(scale.x / scale.z, scale.y / scale.z);
  auto grad = silt::tensor_t<float>(shapeOut, silt::host_t::GPU);
  __grad_centered<<<block(shapeIn.elem(), 512), 512>>>(grad, tensor, shapeIn, gscale);
  gpuErrchk(cudaGetLastError());
  return grad;

}

//
// Slope Magnitude from a Precomputed Gradient
//

__global__ void __grad_magnitude (
  silt::tensor_t<float> tensorOut,     //!< Output Magnitude Field
  const silt::tensor_t<float> tensorIn, //!< Input 2-Component Gradient Field
  const silt::shape shape
){

  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n >= shape.elem()) return;

  const auto view = tensorIn.view<silt::vec2>();
  tensorOut[n] = glm::length(view[n]);

}

silt::tensor_t<float> slope(const silt::tensor_t<float>& gradient){

  const silt::shape shapeIn = gradient.shape();
  const silt::shape shapeOut = silt::shape(shapeIn[0], shapeIn[1]);
  auto out = silt::tensor_t<float>(shapeOut, silt::host_t::GPU);
  __grad_magnitude<<<block(shapeOut.elem(), 512), 512>>>(out, gradient, shapeOut);
  gpuErrchk(cudaGetLastError());
  return out;

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
  
  const int64_t n = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if(n >= shape.elem()) return;
  
  using vec = silt::fvec<D>;
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
  const vec vnn = shape.oob(ipos + silt::ivec2(-1,-1)) ? v00 : viewIn[shape.flatten(ipos + silt::ivec2(-1,-1))];
  const vec vpp = shape.oob(ipos + silt::ivec2( 1, 1)) ? v00 : viewIn[shape.flatten(ipos + silt::ivec2( 1, 1))];
  const vec vpn = shape.oob(ipos + silt::ivec2( 1,-1)) ? v00 : viewIn[shape.flatten(ipos + silt::ivec2( 1,-1))];
  const vec vnp = shape.oob(ipos + silt::ivec2(-1, 1)) ? v00 : viewIn[shape.flatten(ipos + silt::ivec2(-1, 1))];

  float hx = (1.0f / scale.x / scale.x);
  float hy = (1.0f / scale.y / scale.y);

  vec LH = (vn0 - v00)*hx + (vp0 - v00)*hx + (v0n - v00)*hy + (v0p - v00)*hy;
  vec LD = 0.5f*(vnn - v00)*hx + 0.5f*(vpp - v00)*hx + 0.5f*(vpn - v00)*hy + 0.5f*(vnp - v00)*hy;

  viewOut[n] = 0.5f * LH + 0.5f * LD;

}

//! 2D Tensor Laplacian
silt::tensor_t<float> laplacian(const silt::tensor_t<float>& tensor, const silt::vec2 scale) {

  // basically implement the component-wise laplacian function...
  //  should return the exact same dimsensionality of the previous vector.

  const silt::shape shapeIn = tensor.shape();
  const silt::shape shape = silt::shape(shapeIn[0], shapeIn[1]);

  auto laplacian = silt::tensor_t<float>(shapeIn, silt::host_t::GPU);

  if(shapeIn[2] == 1) {
    __laplacian<1><<<block(shape.elem(), 512), 512>>>(laplacian, tensor, shape, scale);
    gpuErrchk(cudaGetLastError());
  }

  if(shapeIn[2] == 2) {
    __laplacian<2><<<block(shape.elem(), 512), 512>>>(laplacian, tensor, shape, scale);
    gpuErrchk(cudaGetLastError());
  }

  return laplacian;

}

} // end of namespace soil
