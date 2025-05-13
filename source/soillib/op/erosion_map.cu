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

//! Nearest Support Point
__device__ int __nearest(const map_t& map, const vec2 pos){

  return map.index.flatten(pos);

}

//! Sample a Position within the Domain
//! associated with scaled probability
template<typename T>
__device__ void __sample(T& part, map_t& map, curandState* rand, const size_t n, const size_t N){

  part.pos = vec2 {
    curand_uniform(rand)*float(map.index[0]),
    curand_uniform(rand)*float(map.index[1])
  };
  part.ind = __nearest(map, part.pos);

  const float P = 1.0f / float(map.index.elem()); // Sampling Probability
  part.Q = P * float(N);                            // Sampling Weight

}

__device__ vec3 __normal(const map_t& map, const vec2 pos, const vec3 scale){

  lerp5_t<float> lerp;
  lerp.gather(map.height, map.sediment, map.index, ivec2(pos));
  const vec2 grad = lerp.grad(scale);
  return glm::normalize(vec3(-grad.x, -grad.y, 1.0f));

}

//! Steepest Surface Direction
__device__ vec2 __steepest(const map_t& map, const vec2 pos, const vec3 scale){

  //  lerp5_t<float> lerp;
  //  lerp.gather(model.height, model.sediment, model.index, pos);
  //  const vec2 grad = lerp.grad(scale);
  //  return glm::normalize(vec3(-grad.x, -grad.y, 1.0f));

  const vec2 shift[8] = {
    vec2(-1.0, -1.0),
    vec2( 0.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  0.0),
    vec2( 1.0,  0.0),
    vec2(-1.0,  1.0),
    vec2( 0.0,  1.0),
    vec2( 1.0,  1.0)
  };

  int mini = -1;
  float minh = map.height[map.index.flatten(pos)] + map.sediment[map.index.flatten(pos)];;
  
  for(int i = 0; i < 8; ++i){
    ivec2 npos = pos + shift[i];
    if(map.index.oob(npos)){
      continue;
    }
    float h = map.height[map.index.flatten(npos)] + map.sediment[map.index.flatten(npos)];
    if(h <= minh){
      mini = i;
      minh = h;
    }
  }
  
  if(mini == -1)
  return vec2(0.0f);
  else return shift[mini];
  
}

__device__ float __hdiff(const map_t& map, const param_t& param, const vec2 pos) {

  const vec3 scale = map.scale * 1E3f;  // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]
  const float Ac = scale.x*scale.y;       // Cell Area [m^2]
  const float Z = Ac * scale.z;           // Height Conversion [m^3]

  // this should just use the momentum direction right?

  const vec2 dir = __steepest(map, pos, scale);
  vec2 npos = vec2(pos) + dir;
  if(map.index.oob(npos)){
    return 0.0f;
  }

  const float dist = glm::length(cl*dir);

  // Stable Bank-Height Computation:

  int find = map.index.flatten(pos);
  int nind = map.index.flatten(npos);

  const float hf_0 = map.height[find];
  const float hn_0 = map.height[nind];
  const float hf_1 = map.sediment[find];
  const float hn_1 = map.sediment[nind];

  const float hf = scale.z * (hf_0 + hf_1);
  const float hn = scale.z * (hn_0 + hn_1);

  const float stable = (hn + param.critSlope*dist);  // [m]
  const float hdiff = hf - stable;
  return hdiff;

}

__device__ float __slope(const map_t& map, const param_t& param, const vec2 pos, const vec2 dir) {

  const vec3 scale = map.scale * 1E3f;    // Cell Scale [m] (conv. from km)
  const vec2 cl = vec2(scale.x, scale.y); // Cell Length [m, m]

  if(glm::length(dir) == 0.0f){
    return 0.0f;
  }

  const vec2 npos = pos + glm::normalize(dir);
  if(npos.x < 0.5f) return -param.exitSlope;
  if(npos.y < 0.5f) return -param.exitSlope;
  if(map.index.oob(npos)){
    return -param.exitSlope;
  }
  
  const int find = map.index.flatten(pos);
  const int nind = map.index.flatten(npos);
  float h0 = (map.height[find] + map.sediment[find])*scale.z;
  float h1 = (map.height[nind] + map.sediment[nind])*scale.z;
  return (h1 - h0)/glm::length(cl);

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