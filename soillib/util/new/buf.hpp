#ifndef SOILLIB_UTIL_BUF_NEW
#define SOILLIB_UTIL_BUF_NEW

#include <soillib/soillib.hpp>
#include <iostream>

namespace soil {
namespace nnn {

//! buf_base is an abstract polymorphic buffer type
//!
//! buf_base allows for specialization of strict-typed
//! buffer types, which implement memory safe operations.
//! 
//! buf_base additionally handles ownership tracking.
//!
struct buf_base {

  buf_base() = default;
  virtual ~buf_base() = default;

  buf_base(buf_base& rhs){
    this->_owns = false;
  }

  buf_base(buf_base&& rhs){
    this->_owns = rhs.owns();
    rhs._owns = false;
  }

  virtual bool owns() const = 0;    //!< Retrieve Memory Ownership Flag

  virtual void allocate(const size_t size)  = 0;  //!< Allocate Memory Buffer
  virtual void deallocate()                 = 0;  //!< De-Allocate Memory Buffer

  virtual void*  data() = 0;        //!< Retrieve Raw Data Pointer
  virtual size_t size() const = 0;  //!< Retrieve Size of Buffer in Bytes
  virtual size_t elem() const = 0;  //!< Retreive Number of Typed Elements

protected:
  bool _owns = false; //!< Raw Data Ownership Flag
};

//! buf_t<T> is a strict-typed, owning raw-data extent.
//! 
template<typename T>
struct buf_t: buf_base {

  buf_t() = default;
  buf_t(const size_t size){
    if(size == 0)
      throw std::invalid_argument("size must be greater than 0");
    this->allocate(size);
  }

  ~buf_t(){
    this->deallocate(); 
  }

  // Allocator / Deallocator

  void allocate(const size_t size){
    if(this->_data != NULL)
      throw std::runtime_error("can't allocate over allocated buffer");
    this->_data = new T[size];
    this->_size = size;
    this->_owns = true;
  }

  void deallocate(){
    if(this->_data == NULL) 
      return;
    if(!this->_owns) 
      return;
    delete[] this->_data;
    this->_size = 0;
    this->_owns = false;
  }

  T& operator[](const size_t index){
    if(index >= this->elem())
      throw std::range_error("index is out of range");
    return *(this->_data + index);
  }

  // Member Function Implementations

  inline bool owns()    const { return this->_owns; }
  inline void* data()         { return (void*)this->_data; }
  inline size_t size()  const { return sizeof(T) * this->_size; }
  inline size_t elem()  const { return this->_size; }

private:
  T* _data = NULL;    //!< Raw Data Pointer Member
  size_t _size = 0;   //!< Data Size in Bytes Member
};

//! buf is a polymorphic dynamically typed buffer container
//!
//! buf contains a strict-typed, memory-safe buffer struct,
//! and provides and interface so that underlying type is
//! runtime constructable for dynamic memory typing.
//!
struct buf {

  buf() = default;

  /*
  template<typename ...Args>
  buf(const char* type, Args&& ...args):
    _buf(make(type, std::forward<Args>(args)...)){}
  */

  ~buf(){
    if(this->_buf != NULL){
      delete this->_buf;
    }
  }

  // Generic Forwarding Factory Function

  template<typename ...Args>
  static buf_base* make(const char* type, Args&& ...args){
    if(type == "int")     return new buf_t<int>(std::forward<Args>(args)...);
    if(type == "float")   return new buf_t<float>(std::forward<Args>(args)...);
    if(type == "double")  return new buf_t<double>(std::forward<Args>(args)...);
    return NULL;
  }

  template<typename ...Args>
  void emplace(const char* type, Args&& ...args){
    if(this->_buf != NULL)
      throw std::runtime_error("buffer is already allocated");
    this->_buf = buf::make(type, std::forward<Args>(args)...);
  }

  //! Strict-Typed Buffer Implementation Retrieval
  template<typename T> buf_t<T> as(){
    buf_t<T> copy = *dynamic_cast<buf_t<T>*>(_buf);
    return copy;
  }

  // Type Overriding

  void allocate(const size_t size){
    this->_buf->allocate(size);
  }

  void deallocate(const size_t size){
    this->_buf->deallocate();
  }

  size_t size() const { 
    return this->_buf->size(); 
  }

  size_t elem() const {
    return this->_buf->elem();
  }

  void* data() { 
    return this->_buf->data(); 
  }

private:
  buf_base* _buf = NULL;
};

}
}

#endif