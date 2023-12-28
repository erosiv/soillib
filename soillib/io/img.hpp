#ifndef SOILLIB_IO_IMG
#define SOILLIB_IO_IMG

#include <soillib/soillib.hpp>
#include <soillib/util/slice.hpp>

#include <functional>

namespace soil {
namespace io {

template<typename T>
struct img {

  typedef T value_t;

  img(){};
  
  img(const uint32_t width, const uint32_t height):
  width{width},height{height}{ 
    allocate();
  }

  img(const glm::ivec2 res):
    img(res.x, res.y){}

  ~img(){
    if(data == NULL)
      return;
    delete[] data;
    data = NULL;
  }

  void allocate(){
    if(data != NULL)
      return;
    data = new value_t[width*height];
  }

  // Fill Method

  void fill(std::function<value_t(const glm::ivec2)> handle){
    for(size_t x = 0; x < width; ++x)
    for(size_t y = 0; y < height; ++y)
      data[y*width + x] = handle(glm::ivec2(x, y));
  }

  // Subscript Operator

  value_t& operator[](const glm::ivec2 pos){
    return data[pos.y * width + pos.x];
  }

  // Size Retrieval
  const inline glm::ivec2 dim() const {
    return {width, height};
  }

protected:
  value_t* data = NULL;
  uint32_t width = 0;
  uint32_t height = 0;
};

}; // end of namespace io
}; // end of namespace soil

#endif