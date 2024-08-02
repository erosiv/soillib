#ifndef SOILLIB_UTIL_BUF_NEW
#define SOILLIB_UTIL_BUF_NEW

#include <memory>
#include <iostream>

#include <soillib/soillib.hpp>
#include <soillib/util/shape.hpp>

namespace soil {

template<typename T> struct array_t;

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

  virtual void*  data() = 0;        //!< Retrieve Raw Data Pointer
  virtual size_t size() const = 0;  //!< Retrieve Size of Buffer in Bytes
  virtual size_t elem() const = 0;  //!< Retreive Number of Typed Elements

protected:
  virtual void allocate(const size_t size)  = 0;  //!< Allocate Memory Buffer
  virtual void deallocate()                 = 0;  //!< De-Allocate Memory Buffer

};

//! buf_t<T> is a strict-typed, owning raw-data extent.
//! 
template<typename T>
struct array_t: array_b {

  // Constructors / Destructor

  array_t() = default;
  array_t(const size_t _size){
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
    this->_size = size;
  }

  void deallocate(){
    this->_data = NULL;
    this->_size = 0;
  }

  // Member Function Implementations

  inline void fill(const T value){
    for(size_t i = 0; i < this->_size; ++i)
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

  inline void* data()         { return (void*)this->_data.get(); }
  inline size_t size()  const { return sizeof(T) * this->_size; }
  inline size_t elem()  const { return this->_size; }

private:
  std::shared_ptr<T[]> _data = NULL;  //!< Raw Data Pointer Member 
  size_t _size = 0;                   //!< Data Size in Bytes Member
};

struct array {

  array(std::string type, std::vector<size_t> v):
    _shape(v){
    _array = std::shared_ptr<array_b>(make(type, _shape.elem()));
  }

  // Factory Function Implementation

  template<typename ...Args>
  static array_b* make(std::string type, Args&& ...args){
    if(type == "int")     return new array_t<int>(std::forward<Args>(args)...);
    if(type == "float")   return new array_t<float>(std::forward<Args>(args)...);
    if(type == "double")  return new array_t<double>(std::forward<Args>(args)...);
    throw std::invalid_argument("invalid argument for type");
  }

  // Virtual Interface Implementation

  inline void* data() { return this->_array->data(); }
  inline size_t size() const { return this->_array->size(); }
  inline size_t elem() const { return this->_array->elem(); }

  // Shape Related Methods

  inline soil::shape shape() { return this->_shape; };

  // Casting / Re-Interpretation

  //! Strict-Typed Buffer Implementation Retrieval
  template<typename T> array_t<T> as(){
    return *dynamic_cast<array_t<T>*>(this->_array.get());
  }

private:
  std::shared_ptr<array_b> _array;
  soil::shape _shape;
};

}

#endif