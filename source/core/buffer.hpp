#ifndef SOILLIB_BUFFER
#define SOILLIB_BUFFER

//! A buffer represents a raw data extent
//! \todo add more detail about this file

#include <soillib/soillib.hpp>
#include <soillib/core/types.hpp>

namespace soil {

//! \todo Make sure that buffers are "re-interpretable"!

//! buffer_t<T> is a strict-typed, raw-data extent.
//! 
template<typename T>
struct buffer_t: typedbase {

  buffer_t():
    _data{NULL},_size{0}{}

  buffer_t(const size_t size){ 
    this->allocate(size); 
  }
  
  ~buffer_t() override { 
    this->deallocate(); 
  }

  void allocate(const size_t size); //!< Allocate Raw Data Buffer
  void deallocate();                //!< De-Allocate Raw Data Buffer

  //! Const Subscript Operator
  T operator[](const size_t index) const noexcept {
    return this->_data[index];
  }

  //! Non-Const Subscript Operator
  T& operator[](const size_t index) noexcept {
    return this->_data[index];
  }

  //! Type Enumerator Retrieval
  constexpr soil::dtype type() noexcept override { 
    return soil::typedesc<T>::type; 
  }

  inline size_t elem()  const { return this->_size; }               //!< Number of Elements
  inline size_t size()  const { return this->elem() * sizeof(T); }  //!< Total Size in Bytes
  inline void* data()         { return (void*)this->_data.get(); }  //!< Raw Data Pointer

private:
  std::shared_ptr<T[]> _data; //!< Raw Data Pointer Member 
  size_t _size;               //!< Number of Data Elements
};

template<typename T>
void buffer_t<T>::allocate(const size_t size){
  if(this->_data != NULL) throw std::runtime_error("can't allocate over allocated buffer");
  if(size == 0)           throw std::invalid_argument("size must be greater than 0");
  this->_data = std::make_shared<T[]>(size);
  this->_size = size;
}

template<typename T>
void buffer_t<T>::deallocate(){ 
  this->_data = NULL;
  this->_size = 0;
}

//
//
//

//! Array variant wrapper type: Implements visitors interface...
struct buffer {

  buffer(){}

  //! \todo FIX THE VALUE SEMANTICS OF ARRAY AND IMPLS!

  // Existing Instance: Hold Reference
  template<typename T>
  buffer(soil::buffer_t<T>& buf):
    impl(&buf){}

  // New Instance: Create new Holder
  template<typename T>
  buffer(soil::buffer_t<T>&& buf){
    impl = new soil::buffer_t<T>(buf);
  }

  buffer(const soil::dtype type, const size_t size):
    impl{make(type, size)}{}

  //! retrieve the strict-typed type enumerator
  inline soil::dtype type() const noexcept {
    return this->impl->type();
  }

  //! unsafe cast to strict-type
  template<typename T> inline buffer_t<T>& as() noexcept {
    return static_cast<buffer_t<T>&>(*(this->impl));
  }

  template<typename T> inline const buffer_t<T>& as() const noexcept {
    return static_cast<buffer_t<T>&>(*(this->impl));
  }
  
  //! Const Subscript Operator
  template<typename T> T operator[](const size_t index) const {
    return typeselect(this->type(), [self=this, index]<typename S>(){
      return self->as<S>().template operator[](index);
    });
  }

  //! Non-Const Subscript Operator
  template<typename T> T& operator[](const size_t index) {
    return typeselect(this->type(), [self=this, index]<typename S>(){
      return self->as<S>().template operator[](index);
    });
  }

  // Data Inspection Operations (Type-Deducing)
  
  size_t elem() const {
    return typeselect(this->type(), [self=this]<typename S>(){
      return self->as<S>().elem();
    });
  }

  size_t size() const {
    return typeselect(this->type(), [self=this]<typename S>(){
      return self->as<S>().size();
    });
  }

  void* data() {
    return typeselect(this->type(), [self=this]<typename S>(){
      return self->as<S>().data();
    });
  }

private:

  static typedbase* make(const soil::dtype type, const size_t size) {
    return typeselect(type, [size]<typename S>() -> typedbase* {
      return new soil::buffer_t<S>(size);
    });
  }

  typedbase* impl;  //!< Strict-Typed Implementation Pointer
};

} // end of namespace soil

#endif