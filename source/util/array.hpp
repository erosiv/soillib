#ifndef SOILLIB_UTIL_ARRAY
#define SOILLIB_UTIL_ARRAY

#include <memory>
#include <iostream>

#include <soillib/soillib.hpp>
#include <soillib/util/shape.hpp>
#include <soillib/util/types.hpp>

namespace soil {

//! array_t<T> is a strict-typed, owning raw-data extent.
//! 
template<typename T>
struct array_t {

  array_t() = default;
  array_t(const soil::shape& shape){
    this->allocate(shape);
  }

  ~array_t(){
    this->deallocate(); 
  }

  // Allocator / Deallocator

  void allocate(const soil::shape& shape){
    if(this->_data != NULL)
      throw std::runtime_error("can't allocate over allocated buffer");
    if(shape.elem() == 0)
      throw std::invalid_argument("size must be greater than 0");
    this->_data = std::make_shared<T[]>(shape.elem());
    this->_shape = shape;
  }

  void deallocate(){
    this->_data = NULL;
  }

  // Member Function Implementations

  inline void fill(const T value){
    for(size_t i = 0; i < this->elem(); ++i)
      this->_data[i] = value;
  }

  inline void zero(){
    this->fill(T{0});
  }

  inline void reshape(const soil::shape& shape){
    if(this->elem() != shape.elem())
      throw std::invalid_argument("can't broadcast current shape to new shape");
    else this->_shape = shape;
  }

  // Subscript Operator

  T& operator[](const size_t index){
    if(index >= this->elem())
      throw std::range_error("index is out of range");
    return this->_data[index];
  }

  T operator[](const size_t index) const {
    if(index >= this->elem())
      throw std::range_error("index is out of range");
    return this->_data[index];
  }

  // Data Inspection Member Functions

  inline const char* type() const { return typedesc<T>::name; }

  inline soil::shape shape() const { return this->_shape; }
  inline size_t elem()  const { return this->_shape.elem(); }
  inline size_t size()  const { return this->elem() * sizeof(T); }
  inline void* data()         { return (void*)this->_data.get(); }

private:
  std::shared_ptr<T[]> _data = NULL;  //!< Raw Data Pointer Member 
  soil::shape _shape;
};

using array = std::variant<
  soil::array_t<int>,
  soil::array_t<float>,
  soil::array_t<double>,
  soil::array_t<fvec2>, 
  soil::array_t<fvec3>
>;

} // end of namespace soil

#endif