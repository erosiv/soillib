#ifndef SOILLIB_UTIL_BUF_NEW
#define SOILLIB_UTIL_BUF_NEW

#include <memory>
#include <iostream>

#include <soillib/soillib.hpp>
#include <soillib/util/shape.hpp>
#include <soillib/util/types.hpp>

namespace soil {

//! array_b is an abstract polymorphic array base type,
//! which handles ownership tracking by move semantics.
//!
//! array allows for specialization of strict-typed
//! array types, which implement memory safe operations.
//!

//! array_t<T> is a strict-typed, owning raw-data extent.
//! 
template<typename T>
struct array_t {

  // Constructors / Destructor

  array_t() = default;
  array_t(soil::shape _shape):
    _shape{_shape}{
      auto _size = this->elem();
      if(_size == 0)
        throw std::invalid_argument("size must be greater than 0");
      this->allocate(_size);
    }

  ~array_t(){
    this->deallocate(); 
  }

  // Allocator / Deallocator

  void allocate(const size_t size){
    if(this->_data != NULL)
      throw std::runtime_error("can't allocate over allocated buffer");
    this->_data = std::make_shared<T[]>(size);
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

  inline size_t elem() const {
    return std::visit([](const auto&& shape){
      return shape.elem();
    }, this->shape());
  }

  inline const char* type() const { return typedesc<T>::name; }

  inline soil::shape shape() const { return this->_shape; }

  inline size_t size()  const { return this->elem() * sizeof(T); }
  inline void* data()         { return (void*)this->_data.get(); }

  inline void reshape(soil::shape shape) {
    auto elem = std::visit([](auto&& shape){
      return shape.elem();
    }, shape);
    if(this->elem() != elem)
      throw std::invalid_argument("can't broadcast current shape to new shape");
    else this->_shape = shape;
  }

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