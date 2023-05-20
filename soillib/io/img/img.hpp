#ifndef SOILLIB_IO_IMAGE
#define SOILLIB_IO_IMAGE

#include <soillib/soillib.hpp>
#include <functional>

namespace soil {
namespace io {

template<typename T>
struct img {

  size_t width = 0;
  size_t height = 0;
  T* data = NULL;

  void allocate(){
    if(data != NULL)
      return;
    data = new T[width*height];
  }

  ~img(){
    if(data == NULL)
      return;
    delete[] data;
  }

  bool read(const char* filename);
  bool write(const char* filename);

  // Fill Method

  void fill(std::function<T(const glm::ivec2)> handle){
    for(size_t x = 0; x < width; x++)
    for(size_t y = 0; y < height; y++)
      data[y*width + x] = handle(glm::ivec2(x, y));
  }

  // Subscript Operator

  const T operator[](const glm::ivec2 pos){
    return data[pos.y * width + pos.x];
  }

};

}; // end of namespace io
}; // end of namespace soil

#endif