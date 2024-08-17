#ifndef SOILLIB_MATRIX
#define SOILLIB_MATRIX

namespace soil {

/*
A Matrix is effectively some representation of a soil mixture.
I make this explicitly strict-typed.
Other quantities of the matrix are somehow derived where necessary?

We will deal with mixture-type matrices later, or alternatively
"segmented" matrices, i.e. storage of an index type, later!

Instead of having these properties "settling" and "maxdiff",
we will convert this to a pure data type and have a layer which
does the computation model for those quantities based on the matrix.
*/

namespace matrix {

struct singular {

  singular(){}

  const singular operator+(const singular rhs) const { return *this; }
  const singular operator/(const float rhs) const { return *this; }
  const singular operator*(const float rhs) const { return *this; }

  /*
  // Concept Implementations

  const float maxdiff() const noexcept {
    return 0.8f;
  }

  const float settling() const noexcept {
    return 1.0f;
  }
  */

};

} // end of namespace matrix
} // end of namespace soil

#endif