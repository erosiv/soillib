#ifndef SOILLIB_OP_GATHER
#define SOILLIB_OP_GATHER

#include <soillib/core/shape.hpp>
#include <soillib/core/tensor.hpp>
#include <soillib/core/types.hpp>

#include <math_constants.h>

namespace soil {

template<std::floating_point T>
struct lerp5_t {

  struct sample_t {
    T value = 0.0f;
    bool oob = true;
  };

  // Single-Value Gather Operation:
  //  Note that we can replace this generally with some structure
  //  that performs a sum over multiple values somewhere. For now,
  //  we will just implement two separate functions.

  GPU_ENABLE void gather(const soil::tensor_t<T> &tensor, glm::ivec2 p) {

    const soil::shape shape = tensor.shape();

    for (int i = 0; i < 5; ++i) {
      const glm::ivec2 pos_x = p + glm::ivec2(-2 + i, 0);
      if (!shape.oob(pos_x)) {
        this->x[i].oob = false;
        const size_t ind = shape.flatten(pos_x);
        this->x[i].value = tensor[ind];
      }

      const glm::ivec2 pos_y = p + glm::ivec2(0, -2 + i);
      if (!shape.oob(pos_y)) {
        this->y[i].oob = false;
        const size_t ind = shape.flatten(pos_y);
        this->y[i].value = tensor[ind];
      }
    }
  }

  lerp5_t(){}
  lerp5_t(const soil::tensor_t<T> &tensor, glm::ivec2 p){
    this->gather(tensor, p);
  }

  GPU_ENABLE vec2 grad(const vec3 scale = vec3(1.0f)) const {

    vec2 g = vec2(0, 0);
    const vec2 s = vec2(scale.z / scale.x, scale.z / scale.y);

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

    return g * s; // Pixel Scale to World-Scale
  }

private:
  sample_t x[5];
  sample_t y[5];
};

namespace {

//! Note: For lerp oob, we have to reduce the bound
//! by one so that we have lerp support on the other end.
GPU_ENABLE bool oob(const vec2 pos, const shape shape) {
  if (pos.x < 0)
    return true;
  if (pos.y < 0)
    return true;
  if (pos.x >= shape[0] - 1)
    return true;
  if (pos.y >= shape[1] - 1)
    return true;
  return false;
}

} // namespace

template<typename T>
struct lerp_t {

  typedef T val_t;
  typedef glm::vec<2, T> grad_t;

  GPU_ENABLE lerp_t(T v): v00(v),
                          v01(v),
                          v10(v),
                          v11(v),
                          w1(1), w0(0) {}

  GPU_ENABLE lerp_t(T v00, T v01, T v10, T v11, vec2 w1): v00(v00),
                                                          v01(v01),
                                                          v10(v10),
                                                          v11(v11),
                                                          w1(w1), w0(vec2(1) - w1) {}

  GPU_ENABLE val_t val() const {
    T v{0};
    v += T(w0.x) * T(w0.y) * v00;
    v += T(w0.x) * T(w1.y) * v01;
    v += T(w1.x) * T(w0.y) * v10;
    v += T(w1.x) * T(w1.y) * v11;
    return v;
  }

  GPU_ENABLE grad_t grad() const {
    glm::vec<2, T> v{0};
    v.x -= T(w0.y) * v00;
    v.x -= T(w1.y) * v01;
    v.x += T(w0.y) * v10;
    v.x += T(w1.y) * v11;
    v.y -= T(w0.x) * v00;
    v.y += T(w0.x) * v01;
    v.y -= T(w1.x) * v10;
    v.y += T(w1.x) * v11;
    return v;
  }

private:
  T v00, v01, v10, v11;
  vec2 w0, w1;
};

/*
template<typename T>
GPU_ENABLE lerp_t<T> gather(const soil::`fer_t<T> &buf, const shape shape, vec2 pos) {

ivec2 p00 = ivec2(pos) + ivec2(0, 0);
ivec2 p01 = ivec2(pos) + ivec2(0, 1);
ivec2 p10 = ivec2(pos) + ivec2(1, 0);
ivec2 p11 = ivec2(pos) + ivec2(1, 1);

//  if(oob(p00, index)) return lerp_t<T>(T{CUDART_NAN_F});
//  if(oob(p01, index)) return lerp_t<T>(T{CUDART_NAN_F});
//  if(oob(p10, index)) return lerp_t<T>(T{CUDART_NAN_F});
//  if(oob(p11, index)) return lerp_t<T>(T{CUDART_NAN_F});

int i00 = shape.flatten(p00);
int i01 = shape.flatten(p01);
int i10 = shape.flatten(p10);
int i11 = shape.flatten(p11);

T v00 = buf[i00];
T v01 = buf[i01];
T v10 = buf[i10];
T v11 = buf[i11];

const vec2 w = pos - glm::floor(pos); // vec2(pos.x - floor(pos.x), pos.y - floor(pos.y));
return lerp_t<T>{v00, v01, v10, v11, w};
}
*/

} // end of namespace soil

#endif