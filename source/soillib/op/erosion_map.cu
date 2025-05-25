#ifndef SOILLIB_OP_EROSION_MAP_CU
#define SOILLIB_OP_EROSION_MAP_CU
#define HAS_CUDA

#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <soillib/op/erosion.hpp>
#include <soillib/op/gather.hpp>

#include <soillib/index/kdtree.hpp>
#include <cukd/builder.h>
#include <cukd/knn.h>

//! Map-Based Geometric Functions
//! Note that these should be dependent on the
//! nature of the map / model.

namespace soil {

//! Nearest Support Point
__device__ int __nearest(const map_grid& map, const vec2 pos){
  return map.index.flatten(pos);
}

__device__ float __height(const map_grid& map, const vec2 pos, const vec3 scale) {
  
  if(map.index.oob(pos))
    return CUDART_NAN_F;
  
  int find = map.index.flatten(pos);
  const float hf_0 = map.height[find];
  const float hf_1 = map.sediment[find];
  return (hf_0 + hf_1) * scale.z;

}

__device__ vec2 __grad(const map_grid& map, const vec2 pos, const vec3 scale){

//  lerp5_t<float> lerp;
//  lerp.gather(map.height, map.sediment, map.index, ivec2(pos));
//  return lerp.grad(scale);

  const float h = __height(map, pos, scale);
  const float hn0 = __height(map, pos + vec2(-1, 0), scale);
  const float hp0 = __height(map, pos + vec2( 1, 0), scale);
  const float h0n = __height(map, pos + vec2( 0,-1), scale);
  const float h0p = __height(map, pos + vec2( 0, 1), scale);

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

  return vec2(gx, gy);

}

//
// Derived Quantities
//

template<typename Map>
__device__ vec3 __normal(const Map& map, const vec2 pos, const vec3 scale){

  const vec2 grad = __grad(map, pos, scale);
  return glm::normalize(vec3(-grad.x, -grad.y, 1.0f));

}

template<typename Map>
__device__ float __hdiff(const Map& map, const param_t& param, const vec2 pos) {

  const vec3 scale = map.scale * 1E3f;    // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  vec2 dir = glm::normalize(__grad(map, pos, scale));
  vec2 npos = vec2(pos) - dir;

  const float hf = __height(map, pos, scale);
  const float hn = __height(map, npos, scale);
  if(__isnanf(hf) || __isnanf(hn))
    return 0.0f;

  // Stable Bank-Height Computation:
  const float dist = glm::length(cl*dir);
  const float stable = (hn + param.critSlope*dist);  // [m]
  const float hdiff = hf - stable;
  return hdiff;

}

template<typename Map>
__device__ float __slope(const Map& map, const param_t& param, const vec2 pos, const vec2 dir) {

  if(glm::length(dir) == 0.0f){
    return 0.0f;
  }
  
  const vec2 npos = pos + glm::normalize(dir);
  if(npos.x < 0.5f) return -param.exitSlope;
  if(npos.y < 0.5f) return -param.exitSlope;
  
  const vec3 scale = map.scale * 1E3f;    // Cell Scale [m] (conv. from km)
  const float hf = __height(map, pos, scale);
  const float hn = __height(map, npos, scale);
  if(__isnanf(hf) || __isnanf(hn))
    return -param.exitSlope;
  
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  return (hn - hf)/glm::length(cl);

}

__device__ vec2 __avespeed(const vec2 momentum, const float discharge){
  if(discharge < 1.0f)
    return vec2(0.0f);
  return momentum / discharge;
}

template<typename Map>
__device__ bool __oob(const Map& map, const vec2 pos){
  return map.index.oob(pos);
}

__device__ vec2 __topos(const map_grid& map, const int nearest){
  return map.index.unflatten(nearest);
}

//! Sample a Position within the Domain
//! associated with scaled probability
template<typename T, typename Map>
__device__ void __sample(T& part, Map& map, const size_t n, const size_t N){

  part.pos = vec2 {
    curand_uniform(&map.rand[n])*float(map.index[0]),
    curand_uniform(&map.rand[n])*float(map.index[1])
  };
  part.ind = __nearest(map, part.pos);

  const float P = 1.0f / float(map.elem); // Sampling Probability
  part.Q = P * float(N);                  // Sampling Weight

}

__device__ void __transfer(map_grid& map, const vec2 pos, float transfer, const float Z) {

  // Single-Material Transfer
  const int n = __nearest(map, pos);
  // map.height[n] += transfer / Z;

  // Multi-Material Mass-Transfer
  if(transfer >= 0.0f){

    map.sediment[n] += transfer / Z;

  } else {

    const float maxtransfer = map.sediment[n] * Z;
    float t1 = transfer;
    if(t1 < -maxtransfer){
      t1 = -maxtransfer;
    }

    map.sediment[n] += t1 / Z;

    transfer -= t1;
    map.height[n] += transfer / Z;

  }

}

} // end of namespace soil

#endif