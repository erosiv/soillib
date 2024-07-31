#ifndef SOILLIB_UTIL_BUF_NEW
#define SOILLIB_UTIL_BUF_NEW

#include <memory>
#include <iostream>

#include <soillib/soillib.hpp>
#include <soillib/util/new/shape.hpp>

namespace soil {

template<typename T> struct buf_t;

//! buffer is an abstract polymorphic buffer type,
//! which handles ownership tracking by move semantics.
//!
//! buffer allows for specialization of strict-typed
//! buffer types, which implement memory safe operations.
//!
struct buffer {

  buffer() = default;
  virtual ~buffer() = default;

  // Factory Constructor

  template<typename ...Args>
  buffer(std::string type, Args&& ...args){
    *this = make(type, std::forward<Args>(args)...);
  }

  template<typename ...Args>
  static buffer* make(std::string type, Args&& ...args);

  // Virtual Interface

  virtual void allocate(const size_t size)  = 0;  //!< Allocate Memory Buffer
  virtual void deallocate()                 = 0;  //!< De-Allocate Memory Buffer

  virtual void*  data() = 0;        //!< Retrieve Raw Data Pointer
  virtual size_t size() const = 0;  //!< Retrieve Size of Buffer in Bytes
  virtual size_t elem() const = 0;  //!< Retreive Number of Typed Elements

  virtual soil::shape* shape() = 0;

  //! Strict-Typed Buffer Implementation Retrieval
  template<typename T> buf_t<T> as(){
    return *dynamic_cast<buf_t<T>*>(this);
  }

};

//! buf_t<T> is a strict-typed, owning raw-data extent.
//! 
template<typename T>
struct buf_t: buffer {

  // Constructors / Destructor

  buf_t() = default;

  buf_t(std::vector<size_t> v){
    
    this->_shape = std::shared_ptr<soil::shape>(soil::shape::make(v));

    std::cout<<this->_shape->dims()<<std::endl;

    const size_t size = this->_shape->elem();
    if(size == 0)
      throw std::invalid_argument("size must be greater than 0");
    this->allocate(size);
  }

  /*
  template<typename ...Args>
  buf_t(Args&& ...args){

  }
  */

  buf_t(buf_t& rhs){
    this->_data = rhs._data;
    this->_size = rhs._size;
    this->_shape = rhs._shape;
  }

  buf_t(buf_t&& rhs){
    this->_data = rhs._data;
    this->_size = rhs._size;
    this->_shape = rhs._shape;
  }

  ~buf_t(){
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
    this->_shape = NULL;
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

  soil::shape* shape() {
    return this->_shape.get();
  };

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
  std::shared_ptr<soil::shape> _shape;
  //soil::shape* _shape = NULL;
  std::shared_ptr<T[]> _data = NULL;          //!< Raw Data Pointer Member 
  size_t _size = 0;                           //!< Data Size in Bytes Member
};

// Factory Function Implementation

template<typename ...Args>
buffer* buffer::make(std::string type, Args&& ...args){
  if(type == "int")     return new buf_t<int>(std::forward<Args>(args)...);
  if(type == "float")   return new buf_t<float>(std::forward<Args>(args)...);
  if(type == "double")  return new buf_t<double>(std::forward<Args>(args)...);
  throw std::invalid_argument("invalid argument for type");
}

}

#endif