#ifndef SOILLIB_SAMPLE
#define SOILLIB_SAMPLE

#include <silt/silt.hpp>
#include <silt/core/tensor.hpp>
#include <math_constants.h>

namespace {

template<typename T>
GPU_ENABLE T dot(glm::vec2 a, glm::vec<2,T> b){
  return a.x*b.x + a.y*b.y;
}

template<typename T>
GPU_ENABLE T dot(glm::vec4 a, glm::vec<4,T> b){
  return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

}

//! sample_t is a multi-dimensional interpolation struct,
//! supporting multi-order value and gradient sampling.
//!
template<typename T, size_t D, size_t O>
struct sample_t;

//
// Linear Interpolation
//

template<typename T>
struct sample_t<T, 1, 1> {

  static constexpr size_t Dim = 1;
  static constexpr size_t Order = 1;

  typedef T val_t;
  typedef T grad_t;
  typedef glm::vec<2,T> vec_t;

  GPU_ENABLE sample_t(T v):
    v(v),t(0){}

  GPU_ENABLE sample_t(vec_t v, float t):
    v(v),t(t){}

  GPU_ENABLE val_t val() const {
    return dot<T>(M()*glm::vec2(1.0, t), v);
  }

  GPU_ENABLE grad_t grad() const {
    return dot<T>(M()*glm::vec2(0.0, 1.0), v);
  }

  static constexpr glm::mat2 M(){
    return {
      1.0,  0.0,
      -1.0,  1.0
    };
  }

  vec_t v;
private:
  float t;
};

//
// 2D Lerp
//

template<typename T>
struct sample_t<T, 2, 1> {

  static constexpr size_t Dim = 2;
  static constexpr size_t Order = 1;

  typedef T val_t;
  typedef glm::vec<2, T> grad_t;
  typedef glm::vec<2, T> vec_t;

  GPU_ENABLE sample_t(T v):
    l0(v),
    l1(v),
    t(0){}

  GPU_ENABLE sample_t(vec_t v0, vec_t v1, silt::vec2 w):
    l0(v0, w.y),
    l1(v1, w.y),
    t(w.x){}

  GPU_ENABLE val_t val() const {
    return dot<T>(M()*glm::vec2(1.0, t), vec_t(l0.val(), l1.val()));
  }

  GPU_ENABLE grad_t grad() const {
    grad_t v{0};
    v.x = dot<T>(M()*glm::vec2(0.0, 1.0), vec_t(l0.val(), l1.val())); // grad(val)
    v.y = dot<T>(M()*glm::vec2(1.0, t), vec_t(l0.grad(), l1.grad())); // val(grad)
    return v;
  }

  static constexpr glm::mat2 M(){
    return {
      1.0,  0.0,
      -1.0,  1.0
    };
  } 

  __device__ static sample_t<T, 2, 1> gather(const silt::tensor_t<T>& buf, const silt::ivec2 res, const silt::vec2 pos);
  __device__ static sample_t<T, 2, 1> gather(const silt::const_view_t<T>& buf, const silt::ivec2 res, const silt::vec2 pos);
//  __device__ static sample_t<T, 2, 1> gather(cudaTextureObject_t texObj, const silt::ivec2 res, const silt::vec2 pos);

  sample_t<T, 1, 1> l0, l1;
  float t;
};

//
// Gather Operations for Interpolation Support
//

template<typename T>
__device__ sample_t<T, 2, 1> sample_t<T, 2, 1>::gather(const silt::tensor_t<T>& buf, const silt::ivec2 res, const silt::vec2 pos){

  const silt::ivec2 p00 = silt::ivec2(pos) + silt::ivec2(0, 0);
  const silt::ivec2 p01 = silt::ivec2(pos) + silt::ivec2(0, 1);
  const silt::ivec2 p10 = silt::ivec2(pos) + silt::ivec2(1, 0);
  const silt::ivec2 p11 = silt::ivec2(pos) + silt::ivec2(1, 1);
  silt::vec2 w = pos - glm::floor(pos);

  int i00 = (res.y - p00.y - 1) * res.x + p00.x;
  int i01 = (res.y - p01.y - 1) * res.x + p01.x;
  int i10 = (res.y - p10.y - 1) * res.x + p10.x;
  int i11 = (res.y - p11.y - 1) * res.x + p11.x;

  if(pos.x < 0 && pos.y < 0) return sample_t<T, 2, 1>(T{CUDART_NAN_F});
  if(pos.x > res.x-1 && pos.y > res.y-1) return sample_t<T, 2, 1>(T{CUDART_NAN_F});

  if(pos.x + 1 > res.x - 1){ w.x = 0; i10 = 0; i11 = 0; }
  if(pos.y + 1 > res.y - 1){ w.y = 0; i01 = 0; i11 = 0; }

  const T h00 = buf[i00];
  const T h01 = buf[i01];
  const T h10 = buf[i10];
  const T h11 = buf[i11];

  return sample_t<T, 2, 1>{
    {h00, h01}, 
    {h10, h11},
    w
  };
}

template<typename T>
__device__ sample_t<T, 2, 1> sample_t<T, 2, 1>::gather(const silt::const_view_t<T>& view, const silt::ivec2 res, const silt::vec2 pos){

  const silt::ivec2 p00 = silt::ivec2(pos) + silt::ivec2(0, 0);
  const silt::ivec2 p01 = silt::ivec2(pos) + silt::ivec2(0, 1);
  const silt::ivec2 p10 = silt::ivec2(pos) + silt::ivec2(1, 0);
  const silt::ivec2 p11 = silt::ivec2(pos) + silt::ivec2(1, 1);
  silt::vec2 w = pos - glm::floor(pos);

  int i00 = (res.y - p00.y - 1) * res.x + p00.x;
  int i01 = (res.y - p01.y - 1) * res.x + p01.x;
  int i10 = (res.y - p10.y - 1) * res.x + p10.x;
  int i11 = (res.y - p11.y - 1) * res.x + p11.x;

  if(pos.x < 0) return sample_t<T, 2, 1>(T{CUDART_NAN_F});
  if(pos.y < 0) return sample_t<T, 2, 1>(T{CUDART_NAN_F});
  if(pos.x > res.x - 1) return sample_t<T, 2, 1>(T{CUDART_NAN_F});
  if(pos.y > res.y - 1) return sample_t<T, 2, 1>(T{CUDART_NAN_F});

  if(pos.x + 1 > res.x - 1){ w.x = 0; i10 = 0; i11 = 0; }
  if(pos.y + 1 > res.y - 1){ w.y = 0; i01 = 0; i11 = 0; }

  const T h00 = view[i00];
  const T h01 = view[i01];
  const T h10 = view[i10];
  const T h11 = view[i11];

  return sample_t<T, 2, 1>{
    {h00, h01}, 
    {h10, h11},
    w
  };
}

//template<typename T>
//__device__ sample_t<T, 2, 1> sample_t<T, 2, 1>::gather(cudaTextureObject_t texObj, const ivec2 res, const vec2 pos){
//
//  const ivec2 p00 = ivec2(pos) + ivec2(0, 0);
//  const ivec2 p01 = ivec2(pos) + ivec2(0, 1);
//  const ivec2 p10 = ivec2(pos) + ivec2(1, 0);
//  const ivec2 p11 = ivec2(pos) + ivec2(1, 1);
//  vec2 w = pos - glm::floor(pos);
//
//  // Bounds Check -> NaN
//  if(pos.x < 0) return sample_t<T, 2, 1>(T{CUDART_NAN_F});
//  if(pos.y < 0) return sample_t<T, 2, 1>(T{CUDART_NAN_F});
//  if(pos.x >= res.x) return sample_t<T, 2, 1>(T{CUDART_NAN_F});
//  if(pos.y >= res.y) return sample_t<T, 2, 1>(T{CUDART_NAN_F});
//
//  // Note the differing y ordering here...
//  const T h00 = tex2D<T>(texObj, (float)p00.x, res.y - 1 - (float)p00.y);
//  const T h01 = tex2D<T>(texObj, (float)p01.x, res.y - 1 - (float)p01.y);
//  const T h10 = tex2D<T>(texObj, (float)p10.x, res.y - 1 - (float)p10.y);
//  const T h11 = tex2D<T>(texObj, (float)p11.x, res.y - 1 - (float)p11.y);
//
//  return sample_t<T, 2, 1>{
//    {h00, h01}, 
//    {h10, h11},
//    w
//  };
//}

#endif