#ifndef SOILLIB_MATRIX_POROSITY
#define SOILLIB_MATRIX_POROSITY

/*==========================
soillib unary soil matrix:
single soil type descriptor
==========================*/

namespace soil {
namespace matrix {

struct porous {

  float value = 1.0f;

  porous operator+(const porous rhs) {
    return porous(this->value + rhs.value); 
  }

  porous operator/(const float rhs) { 
    return porous(this->value/rhs); 
  }

  porous operator*(const float rhs) { 
    return porous(this->value*rhs); 
  }

};

} // end of namespace matrix
} // end of namespace soil

#endif // SOILLIB_MATRIX_SINGULAR