#ifndef SOILLIB_OP_EROSION_MAP_CU
#define SOILLIB_OP_EROSION_MAP_CU
#define HAS_CUDA

#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <soillib/op/erosion.hpp>
#include <soillib/op/gather.hpp>

//! Map-Based Geometric Functions
//! Note that these should be dependent on the
//! nature of the map / model.

namespace soil {

__device__ vec2 __grad(const map_t& map, const vec2 pos, const vec3 scale){

  lerp5_t<float> lerp;
  lerp.gather(map.height, map.sediment, map.index, ivec2(pos));
  return lerp.grad(scale);

}

__device__ vec3 __normal(const map_t& map, const vec2 pos, const vec3 scale){

  const vec2 grad = __grad(map, pos, scale);
  return glm::normalize(vec3(-grad.x, -grad.y, 1.0f));

}

__device__ float __height(const map_t& map, const vec2 pos, const vec3 scale) {
  
  if(map.index.oob(pos))
    return CUDART_NAN_F;
  
  int find = map.index.flatten(pos);
  const float hf_0 = map.height[find];
  const float hf_1 = map.sediment[find];
  return (hf_0 + hf_1) * scale.z;

}

__device__ float __hdiff(const map_t& map, const param_t& param, const vec2 pos) {

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

__device__ float __slope(const map_t& map, const param_t& param, const vec2 pos, const vec2 dir) {

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

//! Nearest Support Point
__device__ int __nearest(const map_t& map, const vec2 pos){

  return map.index.flatten(pos);

}

//! Sample a Position within the Domain
//! associated with scaled probability
template<typename T>
__device__ void __sample(T& part, map_t& map, const size_t n, const size_t N){

  part.pos = vec2 {
    curand_uniform(&map.rand[n])*float(map.index[0]),
    curand_uniform(&map.rand[n])*float(map.index[1])
  };
  part.ind = __nearest(map, part.pos);

  const float P = 1.0f / float(map.elem); // Sampling Probability
  part.Q = P * float(N);                  // Sampling Weight

}

__device__ void __transfer(map_t& map, const size_t n, float transfer, const float Z) {

  // Single-Material Transfer
//  model.height[n] += transfer / Z;

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