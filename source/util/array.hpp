#ifndef SOILLIB_UTIL_ARRAY
#define SOILLIB_UTIL_ARRAY

#include <memory>
#include <iostream>
#include <variant>

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

  inline soil::dtype type() const { return typedesc<T>::type; }
  inline soil::shape shape() const { return this->_shape; }
  inline size_t elem()  const { return this->_shape.elem(); }
  inline size_t size()  const { return this->elem() * sizeof(T); }
  inline void* data()         { return (void*)this->_data.get(); }

private:
  std::shared_ptr<T[]> _data = NULL;  //!< Raw Data Pointer Member 
  soil::shape _shape;
};

using array_v = soil::multi_t<soil::array_t>;

//! Array variant wrapper type: Implements visitors interface...
struct array {

  array(){}

  template<typename T>
  array(const soil::array_t<T> _array):
    _array(_array){}

  array(const soil::dtype type, const soil::shape& shape):
    _array{make(type, shape)}{}

  auto type() const {
    return std::visit([](auto&& _array) -> soil::dtype {
      return _array.type();
    }, this->_array);
  }

  auto shape() const {
    return std::visit([](auto&& _array) -> soil::shape {
      return _array.shape();
    }, this->_array);
  }

  auto elem() const {
    return std::visit([](auto&& _array) -> size_t {
      return _array.elem();
    }, this->_array);
  }

  auto size() const {
    return std::visit([](auto&& _array) -> size_t {
      return _array.size();
    }, this->_array);
  }

  auto data() {
    return std::visit([](auto&& _array) -> void* {
      return _array.data();
    }, this->_array);
  }

  soil::multi operator[](const size_t index) const {
    return std::visit([&index](auto&& _array) -> soil::multi {
      return _array[index];
    }, this->_array);
  }

  array& zero(){
    std::visit([](auto&& _array){
      _array.zero();
    }, this->_array);
    return *this;
  }

  array& reshape(const soil::shape& shape){
    std::visit([&shape](auto&& _array){
      _array.reshape(shape);
    }, this->_array);
    return *this;
  }

  template<typename T>
  array& fill(const T value){
    auto array = std::get<array_t<T>>(this->_array);
    array.fill(value);
    return *this;
  }

  template<typename T>
  void set(const size_t index, const T value){
    auto array = std::get<array_t<T>>(this->_array);
    array[index] = value;
  }

  template<typename T>
  void set_multi(const size_t index, const soil::multi value){
    auto array = std::get<array_t<T>>(this->_array);
    array[index] = std::visit([](auto&& args){
      return (T)args;
    }, value);
  }

  static array_v make(const soil::dtype type, const soil::shape& shape){
    if(type == soil::INT)     return soil::array_t<int>(shape);
    if(type == soil::FLOAT32) return soil::array_t<float>(shape);
    if(type == soil::FLOAT64) return soil::array_t<double>(shape);
    if(type == soil::VEC2)    return soil::array_t<vec2>(shape);
    throw std::invalid_argument("invalid type argument");
  }

//private:
  array_v _array;
};

} // end of namespace soil

#endif