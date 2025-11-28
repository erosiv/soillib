#ifndef SOILLIB_OP_EROSION_MAP_CU
#define SOILLIB_OP_EROSION_MAP_CU
#define HAS_CUDA

#include <silt/core/types.hpp>
#include <silt/core/tensor.hpp>
#include <silt/op/gather.hpp>

#include <soillib/model/path/erosion.hpp>

__device__ silt::vec2 __grad(
  const silt::tensor_t<float>& height,
  const silt::shape shape,
  const silt::vec3 scale,
  const silt::vec2 pos
){

  const int i00 = shape.flatten(pos + silt::vec2( 0, 0));
  const int in0 = shape.flatten(pos + silt::vec2(-1, 0));
  const int ip0 = shape.flatten(pos + silt::vec2( 1, 0));
  const int i0n = shape.flatten(pos + silt::vec2( 0,-1));
  const int i0p = shape.flatten(pos + silt::vec2( 0, 1));

  const float h = height[i00] * scale.z;
  const float hn0 = shape.oob(pos + silt::vec2(-1, 0)) ? h : height[in0] * scale.z;
  const float hp0 = shape.oob(pos + silt::vec2( 1, 0)) ? h : height[ip0] * scale.z;
  const float h0n = shape.oob(pos + silt::vec2( 0,-1)) ? h : height[i0n] * scale.z;
  const float h0p = shape.oob(pos + silt::vec2( 0, 1)) ? h : height[i0p] * scale.z;

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

  return silt::vec2(gx, gy);

}

__device__ float __slope (
  const silt::tensor_t<float>& height,
  const silt::shape shape,
  const silt::vec3 scale,
  const silt::vec2 pos,
  const float exitSlope
){

  const int i00 = shape.flatten(pos + silt::vec2( 0, 0));
  const int in0 = shape.flatten(pos + silt::vec2(-1, 0));
  const int ip0 = shape.flatten(pos + silt::vec2( 1, 0));
  const int i0n = shape.flatten(pos + silt::vec2( 0,-1));
  const int i0p = shape.flatten(pos + silt::vec2( 0, 1));

  const float h = height[i00] * scale.z;
  const float hn0 = shape.oob(pos + silt::vec2(-1, 0)) ? CUDART_NAN_F : height[in0] * scale.z;
  const float hp0 = shape.oob(pos + silt::vec2( 1, 0)) ? CUDART_NAN_F : height[ip0] * scale.z;
  const float h0n = shape.oob(pos + silt::vec2( 0,-1)) ? CUDART_NAN_F : height[i0n] * scale.z;
  const float h0p = shape.oob(pos + silt::vec2( 0, 1)) ? CUDART_NAN_F : height[i0p] * scale.z;

  // Min Gradient Computation w. Bounds Handling
  float gx = 0.0f;
  if(!__isnanf(hn0) && !__isnanf(hp0)){
    gx = glm::max(gx, (h - hn0)/scale.x);
    gx = glm::max(gx, (h - hp0)/scale.x);
  } else if(__isnanf(hn0)){
    gx = glm::max(gx, (h - hp0)/scale.x);
    gx = glm::max(gx, exitSlope);
  } else if(__isnanf(hp0)){
    gx = glm::max(gx, (h - hn0)/scale.x);
    gx = glm::max(gx, exitSlope);
  }

  float gy = 0.0f;
  if(!__isnanf(h0n) && !__isnanf(h0p)){
    gy = glm::max(gy, (h - h0n)/scale.y);
    gy = glm::max(gy, (h - h0p)/scale.y);
  } else if(__isnanf(h0n)){
    gy = glm::max(gy, (h - h0p)/scale.y);
    gy = glm::max(gy, exitSlope);
  } else if(__isnanf(h0p)){
    gy = glm::max(gy, (h - h0n)/scale.y);
    gy = glm::max(gy, exitSlope);
  }

  // Write to 2D vector view
  return glm::length(silt::vec2(gx, gy));

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