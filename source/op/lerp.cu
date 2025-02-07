#ifndef SOILLIB_LERP_CU
#define SOILLIB_LERP_CU

#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <math_constants.h>

namespace soil {

//
// Scene Sampling and Intersection
//

namespace {

//! Note: For lerp oob, we have to reduce the bound
//! by one so that we have lerp support on the other end.
__device__ bool oob(const vec2 pos, const flat_t<2> index){
  if(pos.x < 0) return true;
  if(pos.y < 0) return true;
  if(pos.x >= index[0]-1) return true;
  if(pos.y >= index[1]-1) return true;
  return false;
}

}

template<typename T>
struct lerp_t {

  typedef T val_t;
  typedef glm::vec<2, T> grad_t;

  __device__ lerp_t(T v):
    v00(v),
    v01(v),
    v10(v),
    v11(v),
    w1(1),w0(0){}

  __device__ lerp_t(T v00, T v01, T v10, T v11, vec2 w1):
    v00(v00),
    v01(v01),
    v10(v10),
    v11(v11),
    w1(w1),w0(vec2(1)-w1){}
  
  __device__ val_t val() const {
    T v{0};
    v += w0.x*w0.y*v00;
    v += w0.x*w1.y*v01;
    v += w1.x*w0.y*v10;
    v += w1.x*w1.y*v11;
    return v;
  }

  __device__ grad_t grad() const {
    glm::vec<2, T> v{0};
    v.x -= w0.y*v00;
    v.x -= w1.y*v01;
    v.x += w0.y*v10;
    v.x += w1.y*v11;
    v.y -= w0.x*v00;
    v.y += w0.x*v01;
    v.y -= w1.x*v10;
    v.y += w1.x*v11;
    return v;
  }

private:
  T v00, v01, v10, v11;
  vec2 w0, w1;
};

template<typename T>
__device__ lerp_t<T> gather(const soil::buffer_t<T>& buf, const soil::flat_t<2> index, vec2 pos){

  ivec2 p00 = ivec2(pos) + ivec2(0, 0);
  ivec2 p01 = ivec2(pos) + ivec2(0, 1);
  ivec2 p10 = ivec2(pos) + ivec2(1, 0);
  ivec2 p11 = ivec2(pos) + ivec2(1, 1);

  if(oob(p00, index)) return lerp_t<T>(T{CUDART_NAN_F});
  if(oob(p01, index)) return lerp_t<T>(T{CUDART_NAN_F});
  if(oob(p10, index)) return lerp_t<T>(T{CUDART_NAN_F});
  if(oob(p11, index)) return lerp_t<T>(T{CUDART_NAN_F});

//  if(index.oob(p00)) return lerp_t<T>(T{CUDART_NAN_F});
//  if(index.oob(p01)) return lerp_t<T>(T{CUDART_NAN_F});
//  if(index.oob(p10)) return lerp_t<T>(T{CUDART_NAN_F});
//  if(index.oob(p11)) return lerp_t<T>(T{CUDART_NAN_F});

  int i00 = index.flatten(p00);
  int i01 = index.flatten(p01);
  int i10 = index.flatten(p10);
  int i11 = index.flatten(p11);

  T v00 = buf[i00];
  T v01 = buf[i01];
  T v10 = buf[i10];
  T v11 = buf[i11];

  const vec2 w = pos - glm::floor(pos);//vec2(pos.x - floor(pos.x), pos.y - floor(pos.y));
  return lerp_t<T>{v00, v01, v10, v11, w};

}

} // end of namespace soil

#endif