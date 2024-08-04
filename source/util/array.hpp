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
struct array_b {

  array_b() = default;
  virtual ~array_b() = default;

  // Virtual Interface

  virtual const char* type() const = 0; //!< Retrieve Type String

  virtual soil::shape shape() = 0;              //!< Retrieve the full Shape Object
  virtual void reshape(soil::shape shape) = 0;  //!< Re-Shape the Shape Object

  virtual size_t elem() const = 0;  //!< Retreive Number of Typed Elements
  virtual size_t size() const = 0;  //!< Retrieve Size of Buffer in Bytes
  virtual void*  data() = 0;        //!< Retrieve Raw Data Pointer

  virtual void zero() = 0;  //!< Fill Array with Zeros
 
protected:
  virtual void allocate(const size_t size)  = 0;  //!< Allocate Memory Buffer
  virtual void deallocate()                 = 0;  //!< De-Allocate Memory Buffer
};

//! buf_t<T> is a strict-typed, owning raw-data extent.
//! 
template<typename T>
struct array_t: array_b {

  // Constructors / Destructor

  array_t(const std::vector<size_t> v):_shape{v}{
    auto _size = _shape.elem();
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
    this->fill(T(0));
  }

  // Subscript Operator

  T& operator[](const size_t index){
    if(index >= this->elem())
      throw std::range_error("index is out of range");
    return this->_data[index];
  }

  // Data Inspection Member Functions

  inline const char* type() const { return typedesc<T>::name; }

  inline soil::shape shape()  { return this->_shape; }
  inline size_t elem()  const { return this->_shape.elem(); }
  inline size_t size()  const { return this->elem() * sizeof(T); }
  inline void* data()         { return (void*)this->_data.get(); }

  inline void reshape(soil::shape shape) {
    if(this->elem() != shape.elem())
      throw std::invalid_argument("can't broadcast current shape to new shape");
    else this->_shape = shape;
  }

private:
  std::shared_ptr<T[]> _data = NULL;  //!< Raw Data Pointer Member 
  soil::shape _shape;
};

struct array {

  array(std::string type, std::vector<size_t> v){
    _array = std::shared_ptr<array_b>(make(type, v));
  }

  // Virtual Interface Implementation

  inline const char* type() const { return this->_array->type(); }

  inline soil::shape shape()  { return this->_array->shape(); }
  inline size_t elem() const  { return this->_array->elem(); }
  inline size_t size() const  { return this->_array->size(); }
  inline void* data()         { return this->_array->data(); }

  inline void reshape(soil::shape shape) { this->_array->reshape(shape); }

  inline void zero() { this->_array->zero(); }

  // Casting / Re-Interpretation

  //! Strict-Typed Buffer Implementation Retrieval
  template<typename T> array_t<T> as(){
    return *dynamic_cast<array_t<T>*>(this->_array.get());
  }

  template<typename T>
  void fill(T value) {
    if(this->type() == "int") this->as<int>().fill((int)value);
    if(this->type() == "float") this->as<float>().fill((float)value);
    if(this->type() == "double") this->as<double>().fill((double)value);
  }

  template<typename T>
  void set(const size_t index, T value){
    if(this->type() == "int") this->as<int>().operator[](index) = (int)value;
    if(this->type() == "float") this->as<float>().operator[](index) = (float)value;
    if(this->type() == "double") this->as<double>().operator[](index) = (double)value;
  }

  // Factory Function Implementation

  template<typename ...Args>
  static array_b* make(std::string type, Args&& ...args){
    if(type == "int")     return new array_t<int>(std::forward<Args>(args)...);
    if(type == "float")   return new array_t<float>(std::forward<Args>(args)...);
    if(type == "double")  return new array_t<double>(std::forward<Args>(args)...);
    throw std::invalid_argument("invalid argument for type");
  }

private:
  std::shared_ptr<array_b> _array;
};

}

#endif