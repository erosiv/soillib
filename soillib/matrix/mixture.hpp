#ifndef SOILLIB_MATRIX_MIXTURE
#define SOILLIB_MATRIX_MIXTURE

/*==========================
soillib unary soil matrix:
single soil type descriptor
==========================*/

namespace soil {
namespace matrix {

struct mixture_config {
  glm::vec3 color;
};

template<size_t N = 2>
struct mixture {
  
  typedef mixture_config config;

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

  glm::vec3 albedo(config& conf){
    glm::vec3 albedo = glm::vec3(0);
    for(size_t n = 0; n < N; n++)
      albedo += this->weight[n]*conf.color[n];
    return albedo;
  }

  // Concept Implementations

  const float maxdiff() const noexcept {
    return 0.8f;
  }

  const float settling() const noexcept {
    return 1.0f;
  }

};

} // end of namespace matrix
} // end of namespace soil


#ifdef SOILLIB_IO_YAML

template<>
struct soil::io::yaml::cast<soil::matrix::mixture_config> {
  static soil::matrix::mixture_config As(soil::io::yaml& node){
    soil::matrix::mixture_config config;
    config.color = node["color"].As<glm::vec3>();
    return config;
  }
};

#endif

#endif // SOILLIB_MATRIX_SINGULAR