#ifndef SOILLIB_MATRIX_SINGULAR
#define SOILLIB_MATRIX_SINGULAR

/*==========================
soillib unary soil matrix:
single soil type descriptor
==========================*/

namespace soil {
namespace matrix {

struct binary {
  
  float mixture = 0.0f;

  binary(){}

  binary operator+(const binary rhs) {
    this->mixture += rhs.mixture;
    return *this; 
  }
  
  binary operator/(const float rhs) { 
    this->mixture /= rhs;
    return *this; 
  }

  binary operator*(const float rhs) {
    this->mixture *= rhs;
    return *this; 
  }

};

} // end of namespace matrix
} // end of namespace soil

#endif // SOILLIB_MATRIX_SINGULAR