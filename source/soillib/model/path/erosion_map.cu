#ifndef SOILLIB_OP_EROSION_MAP_CU
#define SOILLIB_OP_EROSION_MAP_CU
#define HAS_CUDA

#include <silt/core/types.hpp>
#include <silt/core/tensor.hpp>
#include <silt/op/gather.hpp>

#include <soillib/model/path/erosion.hpp>

//! Various Method of Computing Gradients:
//! In general, we want it to vary smoothly,
//! so it is most likely useful if we interpolate
//! somehow. We can either do that using a gradient
//! method and then lerping, or using the analytical
//! gradients of a higher order method.

//! Cell-Center Sampling:
//!   

template<typename T>
__device__ T __divzero (
  const T& a,
  const float b
) {
  return (b == 0.0f) ? T{0} : (a / b);
}

__device__ bool __oob(
  const silt::shape& shape,
  const silt::vec2 pos
) {

  if(pos.x < 0) return true;
  if(pos.y < 0) return true;
  if(pos.x >= shape[0]) return true;
  if(pos.y >= shape[1]) return true;
  return false;

}

__device__ int __flatten (
  const silt::shape& shape,
  const silt::vec2 pos
) {
  return shape.flatten(pos);
}

__device__ float __length (
  silt::vec2 v
) {
  return sqrtf(v.x * v.x + v.y * v.y);
}

__device__ float __ndot (
  silt::vec2 g,
  silt::vec2 v
) {
  return (g.x * v.x + g.y * v.y) / __length(v);
}

__device__ silt::vec2 __glocal (
  const silt::view_t<silt::vec2>& layers,
  const silt::shape shape,
  const silt::vec3 scale,
  const silt::ivec2 ipos,
  const float exitSlope
) {

  const int i00 = shape.flatten(ipos + silt::ivec2( 0, 0));
  const int in0 = shape.flatten(ipos + silt::ivec2(-1, 0));
  const int ip0 = shape.flatten(ipos + silt::ivec2( 1, 0));
  const int i0n = shape.flatten(ipos + silt::ivec2( 0,-1));
  const int i0p = shape.flatten(ipos + silt::ivec2( 0, 1));

  const float h = layers[i00].x;
  const float hn0 = shape.oob(ipos + silt::ivec2(-1, 0)) ? CUDART_NAN_F : layers[in0].x;
  const float hp0 = shape.oob(ipos + silt::ivec2( 1, 0)) ? CUDART_NAN_F : layers[ip0].x;
  const float h0n = shape.oob(ipos + silt::ivec2( 0,-1)) ? CUDART_NAN_F : layers[i0n].x;
  const float h0p = shape.oob(ipos + silt::ivec2( 0, 1)) ? CUDART_NAN_F : layers[i0p].x;

  // Note: fmaxf returns numeric value if one value is NaN.
  // NaN: On boundary, use signed (downhill) exitslope
  //  aN: Not on boundary, clamp to downhill.

  float gxn = (h - hn0) * scale.z / scale.x;
  if(__isnanf(gxn)) gxn = exitSlope;
  else gxn = fmaxf(gxn, 0.0f);
  
  float gyn = (h - h0n) * scale.z / scale.y;
  if(__isnanf(gyn)) gyn = exitSlope;
  else gyn = fmaxf(gyn, 0.0f);
  
  float gxp = (hp0 - h) * scale.z / scale.x;
  if(__isnanf(gxp)) gxp = -exitSlope;
  else gxp = fminf(gxp, 0.0f);
  
  float gyp = (h0p - h) * scale.z / scale.y;
  if(__isnanf(gyp)) gyp = -exitSlope;
  else gyp = fminf(gyp, 0.0f);

  // Choose Steepest
  
  float gx = 0.0f;
  if(abs(gxn) > abs(gx)) gx = gxn;
  if(abs(gxp) > abs(gx)) gx = gxp;
  
  float gy = 0.0f;
  if(abs(gyn) > abs(gy)) gy = gyn;
  if(abs(gyp) > abs(gy)) gy = gyp;

  return silt::vec2(gx, gy);

}

__device__ silt::vec2 __grad (
  const silt::view_t<silt::vec2>& layers,
  const silt::shape shape,
  const silt::vec3 scale,
  silt::vec2 pos,
  const float exitSlope
) {

  // Basic Local Implementation In-Cell:
  const silt::vec2 g00 = __glocal(layers, shape, scale, pos, exitSlope);
  return g00;

//  // Linear Interpolated Implementation:
//  silt::ivec2 p00, p01, p10, p11;
//  silt::vec2 w;
//
//  p00.x = floorf(pos.x - 0.5f);
//  p01.x = floorf(pos.x - 0.5f);
//  p10.x = 1 + floorf(pos.x - 0.5f);
//  p11.x = 1 + floorf(pos.x - 0.5f);
//
//  p00.y = floorf(pos.y - 0.5f);
//  p01.y = 1 + floorf(pos.y - 0.5f);
//  p10.y = floorf(pos.y - 0.5f);
//  p11.y = 1 + floorf(pos.y - 0.5f);
//
//  if(pos.x - 0.5f < 0.0f) {
//    p00.x = 0;
//    p01.x = 0;
//    p10.x = 0;
//    p11.x = 0;
//  }
//  if(pos.y - 0.5f < 0.0f) {
//    p00.y = 0;
//    p01.y = 0;
//    p10.y = 0;
//    p11.y = 0;
//  }
//  if(pos.x - 0.5f >= float(shape[0] - 1)){
//    p00.x = shape[0] - 1;
//    p01.x = shape[0] - 1;
//    p10.x = shape[0] - 1;
//    p11.x = shape[0] - 1;
//  }
//  if(pos.y - 0.5f >= float(shape[1] - 1)){
//    p00.y = shape[1] - 1;
//    p01.y = shape[1] - 1;
//    p10.y = shape[1] - 1;
//    p11.y = shape[1] - 1;
//  }
//
//  w = (pos - 0.5f) - glm::floor(pos - 0.5f);
////  w.x = fminf(fmaxf(w.x, 0.0f), 1.0f);
////  w.y = fminf(fmaxf(w.y, 0.0f), 1.0f);
//  
//  const silt::vec2 g00 = __glocal(height, shape, scale, p00, exitSlope);
//  const silt::vec2 g10 = __glocal(height, shape, scale, p10, exitSlope);
//  const silt::vec2 g01 = __glocal(height, shape, scale, p01, exitSlope);
//  const silt::vec2 g11 = __glocal(height, shape, scale, p11, exitSlope);
//  return g00 * (1.0f - w.x) * (1.0f - w.y) + g01 * (1.0f - w.x) * w.y + g10 * w.x * (1.0f - w.y) + g11 * w.x * w.y;

}

// __device__ float __height(const map_t& map, const vec2 pos, const float scale_z) {
//   
//   if(map.shape.oob(pos))
//     return CUDART_NAN_F;
//   
//   int find = map.shape.flatten(pos);
//   const float hf_0 = map.height[find];
//   const float hf_1 = map.sediment[find];
//   return (hf_0 + hf_1) * scale_z;
// 
// }

// __device__ void __transfer(map_t& map, const vec2 pos, float transfer, const float Z) {
// 
//   // Single-Material Transfer
//   const int n = __nearest(map, pos);
//   // map.height[n] += transfer / Z;
// 
//   // Multi-Material Mass-Transfer
//   if(transfer >= 0.0f){
// 
//     map.sediment[n] += transfer / Z;
// 
//   } else {
// 
//     const float maxtransfer = map.sediment[n] * Z;
//     float t1 = transfer;
//     if(t1 < -maxtransfer){
//       t1 = -maxtransfer;
//     }
// 
//     map.sediment[n] += t1 / Z;
// 
//     transfer -= t1;
//     map.height[n] += transfer / Z;
// 
//   }
// 
// }

#endif