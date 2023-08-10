#ifndef SOILLIB_MATRIX_SINGULAR
#define SOILLIB_MATRIX_SINGULAR

/*==========================
soillib unary soil matrix:
single soil type descriptor
==========================*/

namespace soil {
namespace matrix {

struct singular {

  singular(){}

  const singular operator+(const singular rhs) const { return *this; }
  const singular operator/(const float rhs) const { return *this; }
  const singular operator*(const float rhs) const { return *this; }

  // Concept Implementations

  const float maxdiff() const noexcept {
    return 0.7f;
  }

  const float settling() const noexcept {
    return 1.0f;
  }

};

} // end of namespace matrix
} // end of namespace soil

#endif // SOILLIB_MATRIX_SINGULAR