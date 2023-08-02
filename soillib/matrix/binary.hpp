#ifndef SOILLIB_MATRIX_SINGULAR
#define SOILLIB_MATRIX_SINGULAR

/*==========================
soillib unary soil matrix:
single soil type descriptor
==========================*/

namespace soil {
namespace matrix {

struct binary_config {
  glm::vec3 colorA = glm::vec3(0);
  glm::vec3 colorB = glm::vec3(1);
};

struct binary {
  
  typedef binary_config config;

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

  glm::vec3 albedo(binary_config& conf){
    return glm::mix(conf.colorA, conf.colorB, mixture);
  }

};

} // end of namespace matrix
} // end of namespace soil


#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<soil::matrix::binary_config> {
  static soil::matrix::binary_config As(soil::io::yaml& node){
    soil::matrix::binary_config config;
    config.colorA = node["colorA"].As<glm::vec3>();
    config.colorB = node["colorB"].As<glm::vec3>();
    return config;
  }
};

#endif

#endif // SOILLIB_MATRIX_SINGULAR