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
  const silt::tensor_t<float>& height,
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

  const float h = height[i00];
  const float hn0 = shape.oob(ipos + silt::ivec2(-1, 0)) ? CUDART_NAN_F : height[in0];
  const float hp0 = shape.oob(ipos + silt::ivec2( 1, 0)) ? CUDART_NAN_F : height[ip0];
  const float h0n = shape.oob(ipos + silt::ivec2( 0,-1)) ? CUDART_NAN_F : height[i0n];
  const float h0p = shape.oob(ipos + silt::ivec2( 0, 1)) ? CUDART_NAN_F : height[i0p];

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

/*
__device__ float __slocal (
  const silt::tensor_t<float>& height,
  const silt::shape shape,
  const silt::vec3 scale,
  const silt::ivec2 ipos,
  const float exitSlope
){

  const int i00 = shape.flatten(ipos + silt::ivec2( 0, 0));
  const int in0 = shape.flatten(ipos + silt::ivec2(-1, 0));
  const int ip0 = shape.flatten(ipos + silt::ivec2( 1, 0));
  const int i0n = shape.flatten(ipos + silt::ivec2( 0,-1));
  const int i0p = shape.flatten(ipos + silt::ivec2( 0, 1));

  const float h = height[i00];
  const float hn0 = shape.oob(ipos + silt::ivec2(-1, 0)) ? CUDART_NAN_F : height[in0];
  const float hp0 = shape.oob(ipos + silt::ivec2( 1, 0)) ? CUDART_NAN_F : height[ip0];
  const float h0n = shape.oob(ipos + silt::ivec2( 0,-1)) ? CUDART_NAN_F : height[i0n];
  const float h0p = shape.oob(ipos + silt::ivec2( 0, 1)) ? CUDART_NAN_F : height[i0p];

  // Min Gradient Computation w. Bounds Handling
  const float gxn = (h - hn0) * scale.z / scale.x;
  const float gxp = (h - hp0) * scale.z / scale.x;
  const float gyn = (h - h0n) * scale.z / scale.y;
  const float gyp = (h - h0p) * scale.z / scale.y;
  
  float gx = 0.0f;
  if(!__isnanf(gxn)) gx = glm::max(gx, gxn);
  if(!__isnanf(gxp)) gx = glm::max(gx, gxp);
  if(__isnanf(gxn) || __isnanf(gxp))
    gx = glm::max(gx, exitSlope);

  float gy = 0.0f;
  if(!__isnanf(gyn)) gy = glm::max(gy, gyn);
  if(!__isnanf(gyp)) gy = glm::max(gy, gyp);
  if(__isnanf(gyn) || __isnanf(gyp))
    gy = glm::max(gy, exitSlope);

  // Write to 2D vector view
  return glm::length(silt::vec2(gx, gy));

}
*/

__device__ silt::vec2 __grad (
  const silt::tensor_t<float>& height,
  const silt::shape shape,
  const silt::vec3 scale,
  silt::vec2 pos,
  const float exitSlope
) {

  // Basic Local Implementation In-Cell:
  const silt::vec2 g00 = __glocal(height, shape, scale, pos, exitSlope);
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

__device__ float __slope (
  const silt::tensor_t<float>& height,
  const silt::shape shape,
  const silt::vec3 scale,
  const silt::vec2 pos,
  const float exitSlope
) {

  const silt::vec2 g00 = __glocal(height, shape, scale, silt::ivec2(pos) + silt::ivec2(0, 0), exitSlope);
  return glm::length(g00);

  /*
  silt::ivec2 p00, p01, p10, p11;
  silt::vec2 w;

  p00.x = floorf(pos.x - 0.5f);
  p01.x = floorf(pos.x - 0.5f);
  p10.x = 1 + floorf(pos.x - 0.5f);
  p11.x = 1 + floorf(pos.x - 0.5f);

  p00.y = floorf(pos.y - 0.5f);
  p01.y = 1 + floorf(pos.y - 0.5f);
  p10.y = floorf(pos.y - 0.5f);
  p11.y = 1 + floorf(pos.y - 0.5f);

  w = (pos - 0.5f) - glm::floor(pos - 0.5f);

  if(pos.x < 0.5f) {
    p00.x = 0;
    p01.x = 0;
    p10.x = 0;
    p11.x = 0;
  }
  if(pos.y < 0.5f) {
    p00.y = 0;
    p01.y = 0;
    p10.y = 0;
    p11.y = 0;
  }
  if(pos.x >= float(shape[0]) - 0.5f){
    p00.x = shape[0] - 1;
    p01.x = shape[0] - 1;
    p10.x = shape[0] - 1;
    p11.x = shape[0] - 1;
  }
  if(pos.y >= float(shape[1]) - 0.5f){
    p00.y = shape[1] - 1;
    p01.y = shape[1] - 1;
    p10.y = shape[1] - 1;
    p11.y = shape[1] - 1;
  }

  const silt::vec2 g00 = __glocal(height, shape, scale, p00, exitSlope);
  const silt::vec2 g10 = __glocal(height, shape, scale, p10, exitSlope);
  const silt::vec2 g01 = __glocal(height, shape, scale, p01, exitSlope);
  const silt::vec2 g11 = __glocal(height, shape, scale, p11, exitSlope);
  const silt::vec2 g = g00 * (1.0f - w.x) * (1.0f - w.y) + g01 * (1.0f - w.x) * w.y + g10 * w.x * (1.0f - w.y) + g11 * w.x * w.y;
  return __length(g);
  */

  //  const silt::vec2 g10 = __glocal(height, shape, scale, silt::ivec2(pos) + silt::ivec2(1, 0), exitSlope);
  //  const silt::vec2 g01 = __glocal(height, shape, scale, silt::ivec2(pos) + silt::ivec2(0, 1), exitSlope);
  //  const silt::vec2 g11 = __glocal(height, shape, scale, silt::ivec2(pos) + silt::ivec2(1, 1), exitSlope);
  //  const silt::vec2 g = g00 * (1.0f - w.x) * (1.0f - w.y) + g01 * (1.0f - w.x) * w.y + g10 * w.x * (1.0f - w.y) + g11 * w.x * w.y;
  //  return glm::length(g);

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

// __device__ vec2 __avespeed(const vec2 momentum, const float discharge){
//   if(discharge < 1.0f)
//     return vec2(0.0f);
//   return momentum / discharge;
// }

// template<typename Map>
// __device__ bool __oob(const Map& map, const vec2 pos){
//   return map.shape.oob(pos);
// }

// template<typename T, typename Map>
// __device__ void __sample(T& part, Map& map, const size_t n, const size_t N){
// 
//   part.pos = vec2 {
//     curand_uniform(&map.rand[n])*float(map.shape[0]),
//     curand_uniform(&map.rand[n])*float(map.shape[1])
//   };
//   part.ind = map.shape.flatten(part.pos);
//   const float P = 1.0f / float(map.elem); // Sampling Probability
//   part.Q = P * float(N);                  // Sampling Weight
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