#ifndef SOILLIB_MATRIX
#define SOILLIB_MATRIX

#include <array>

namespace soil {

//! A matrix is a representation of a soil composition,
//! and can contain various information about the soil.

/*
The problem that I have now is that the enum would require
expansion with knowledge of the type existing. This is a problem.
Do I declare it ahead of time? I can't really "create" a buffer
of that type. Unless I say that it's just a float buffer and is
somehow just reinterpreted.
*/

namespace matrix {

//! A singular matrix means that the entirety of all soil
//! has a single, non-distinguishable composition.
//!
struct singular {

  singular() {}

  const singular operator+(const singular rhs) const { return *this; }
  const singular operator/(const float rhs) const { return *this; }
  const singular operator*(const float rhs) const { return *this; }
};

//! A porous matrix means that the soil composition is
//! distinguished by a single floating point quantity,
//! i.e. the porosity, which evolves during the simulation.
//!
struct porous {

  porous(float value): value(value) {}
  porous(): porous(1.0f) {}

  float value;

  const porous operator+(const porous rhs) const {
    return porous(this->value + rhs.value);
  }

  const porous operator/(const float rhs) const {
    return porous(this->value / rhs);
  }

  const porous operator*(const float rhs) const {
    return porous(this->value * rhs);
  }
};

/*
struct mixture_config {
  glm::vec3 color;
  float maxdiff;
};

template<size_t N = 2>
struct mixture {

  typedef std::array<mixture_config, 2> config;
  static config conf;

  float weight[N];

  mixture(){
    for(size_t n = 0; n < N; n++)
      weight[n] = 0;
  }

  mixture<N>& operator=(const mixture<N> rhs) {
    for(size_t n = 0; n < N; n++)
      this->weight[n] = rhs.weight[n];
    return *this;
  }

  mixture<N> operator+(const mixture<N> rhs) {
    for(size_t n = 0; n < N; n++)
      this->weight[n] += rhs.weight[n];
    return *this;
  }

  mixture<N> operator/(const float rhs) {
    for(size_t n = 0; n < N; n++)
      this->weight[n] /= rhs;
    return *this;
  }

  mixture<N> operator*(const float rhs) {
    for(size_t n = 0; n < N; n++)
      this->weight[n] *= rhs;
    return *this;
  }

  // Concept Implementations

  const float maxdiff() const noexcept {
    float maxdiff = 0.0f;
    for(size_t n = 0; n < N; n++)
      maxdiff += this->weight[n]*conf[n].maxdiff;
    return maxdiff;
  }

  const float settling() const noexcept {
    return 1.0f;
  }

};

template<size_t N>
mixture<N>::config mixture<N>::conf;
*/

} // end of namespace matrix
} // end of namespace soil

#endif