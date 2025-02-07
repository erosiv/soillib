#ifndef SOILLIB_OP_GATHER
#define SOILLIB_OP_GATHER

#include <soillib/core/types.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <math_constants.h>

namespace soil {

template<std::floating_point T>
struct lerp5_t {

  struct sample_t {
    T value = 0.0f;
    bool oob = true;
  };

  template<typename I>
  GPU_ENABLE void gather(const soil::buffer_t<T> &buffer_t, const I index, glm::ivec2 p){

    for (int i = 0; i < 5; ++i) {
      const glm::ivec2 pos_x = p + glm::ivec2(-2 + i, 0);
      if (!index.oob(pos_x)) {
        this->x[i].oob = false;
        const size_t ind = index.flatten(pos_x);
        this->x[i].value = buffer_t[ind];
      }

      const glm::ivec2 pos_y = p + glm::ivec2(0, -2 + i);
      if (!index.oob(pos_y)) {
        this->y[i].oob = false;
        const size_t ind = index.flatten(pos_y);
        this->y[i].value = buffer_t[ind];
      }
    }

  }

  GPU_ENABLE glm::vec2 grad() const {

    glm::vec2 g = glm::vec2(0, 0);

    // X-Element
    if (!this->x[0].oob && !this->x[4].oob)
      g.x = (1.0f * this->x[0].value - 8.0f * this->x[1].value + 8.0f * this->x[3].value - 1.0f * this->x[4].value) / 12.0f;

    else if (!this->x[0].oob && !this->x[3].oob)
      g.x = (1.0f * this->x[0].value - 6.0f * this->x[1].value + 3.0f * this->x[2].value + 2.0f * this->x[3].value) / 6.0f;

    else if (!this->x[0].oob && !this->x[2].oob)
      g.x = (1.0f * this->x[0].value - 4.0f * this->x[1].value + 3.0f * this->x[2].value) / 2.0f;

    else if (!this->x[1].oob && !this->x[4].oob)
      g.x = (-2.0f * this->x[1].value - 3.0f * this->x[2].value + 6.0f * this->x[3].value - 1.0f * this->x[4].value) / 6.0f;

    else if (!this->x[2].oob && !this->x[4].oob)
      g.x = (-3.0f * this->x[2].value + 4.0f * this->x[3].value - 1.0f * this->x[4].value) / 2.0f;

    else if (!this->x[1].oob && !this->x[3].oob)
      g.x = (-1.0f * this->x[1].value + 1.0f * this->x[3].value) / 2.0f;

    else if (!this->x[2].oob && !this->x[3].oob)
      g.x = (-1.0f * this->x[2].value + 1.0f * this->x[3].value) / 1.0f;

    else if (!this->x[1].oob && !this->x[2].oob)
      g.x = (-1.0f * this->x[1].value + 1.0f * this->x[2].value) / 1.0f;

    // Y-Element

    if (!this->y[0].oob && !this->y[4].oob)
      g.y = (1.0f * this->y[0].value - 8.0f * this->y[1].value + 8.0f * this->y[3].value - 1.0f * this->y[4].value) / 12.0f;

    else if (!this->y[0].oob && !this->y[3].oob)
      g.y = (1.0f * this->y[0].value - 6.0f * this->y[1].value + 3.0f * this->y[2].value + 2.0f * this->y[3].value) / 6.0f;

    else if (!this->y[0].oob && !this->y[2].oob)
      g.y = (1.0f * this->y[0].value - 4.0f * this->y[1].value + 3.0f * this->y[2].value) / 2.0f;

    else if (!this->y[1].oob && !this->y[4].oob)
      g.y = (-2.0f * this->y[1].value - 3.0f * this->y[2].value + 6.0f * this->y[3].value - 1.0f * this->y[4].value) / 6.0f;

    else if (!this->y[2].oob && !this->y[4].oob)
      g.y = (-3.0f * this->y[2].value + 4.0f * this->y[3].value - 1.0f * this->y[4].value) / 2.0f;

    else if (!this->y[1].oob && !this->y[3].oob)
      g.y = (-1.0f * this->y[1].value + 1.0f * this->y[3].value) / 2.0f;

    else if (!this->y[2].oob && !this->y[3].oob)
      g.y = (-1.0f * this->y[2].value + 1.0f * this->y[3].value) / 1.0f;

    else if (!this->y[1].oob && !this->y[2].oob)
      g.y = (-1.0f * this->y[1].value + 1.0f * this->y[2].value) / 1.0f;

    return g;

  }

private:
  sample_t x[5];
  sample_t y[5];
};

namespace {

//! Note: For lerp oob, we have to reduce the bound
//! by one so that we have lerp support on the other end.
GPU_ENABLE bool oob(const vec2 pos, const flat_t<2> index){
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

  GPU_ENABLE lerp_t(T v):
    v00(v),
    v01(v),
    v10(v),
    v11(v),
    w1(1),w0(0){}

  GPU_ENABLE lerp_t(T v00, T v01, T v10, T v11, vec2 w1):
    v00(v00),
    v01(v01),
    v10(v10),
    v11(v11),
    w1(w1),w0(vec2(1)-w1){}
  
  GPU_ENABLE val_t val() const {
    T v{0};
    v += w0.x*w0.y*v00;
    v += w0.x*w1.y*v01;
    v += w1.x*w0.y*v10;
    v += w1.x*w1.y*v11;
    return v;
  }

  GPU_ENABLE grad_t grad() const {
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
GPU_ENABLE lerp_t<T> gather(const soil::buffer_t<T>& buf, const soil::flat_t<2> index, vec2 pos){

  ivec2 p00 = ivec2(pos) + ivec2(0, 0);
  ivec2 p01 = ivec2(pos) + ivec2(0, 1);
  ivec2 p10 = ivec2(pos) + ivec2(1, 0);
  ivec2 p11 = ivec2(pos) + ivec2(1, 1);

//  if(oob(p00, index)) return lerp_t<T>(T{CUDART_NAN_F});
//  if(oob(p01, index)) return lerp_t<T>(T{CUDART_NAN_F});
//  if(oob(p10, index)) return lerp_t<T>(T{CUDART_NAN_F});
//  if(oob(p11, index)) return lerp_t<T>(T{CUDART_NAN_F});

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