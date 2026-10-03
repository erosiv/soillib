#pragma once

#include <soillib/soillib.hpp>

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

  //! Bilinear gather at a fractional position. res is (W, H); x is dim0 and y
  //! is dim1, which is contiguous. A buffer in any other order is reordered at
  //! the boundary it came from, not sampled through a second index convention.
  __device__ static sample_t<T, 2, 1> gather(const silt::view_t<const T>& buf, const silt::ivec2 res, const silt::vec2 pos);

  sample_t<T, 1, 1> l0, l1;
  float t;
};

//
// Gather Operations for Interpolation Support
//

template<typename T>
__device__ sample_t<T, 2, 1> sample_t<T, 2, 1>::gather(const silt::view_t<const T>& view, const silt::ivec2 res, const silt::vec2 pos){

  const silt::ivec2 p00 = silt::ivec2(pos) + silt::ivec2(0, 0);
  const silt::ivec2 p01 = silt::ivec2(pos) + silt::ivec2(0, 1);
  const silt::ivec2 p10 = silt::ivec2(pos) + silt::ivec2(1, 0);
  const silt::ivec2 p11 = silt::ivec2(pos) + silt::ivec2(1, 1);
  silt::vec2 w = pos - glm::floor(pos);

  int i00 = p00.x * res.y + p00.y;
  int i01 = p01.x * res.y + p01.y;
  int i10 = p10.x * res.y + p10.y;
  int i11 = p11.x * res.y + p11.y;

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

