#ifndef SOILLIB_BUFFER
#define SOILLIB_BUFFER

//! A buffer represents a raw data extent
//! \todo add more detail about this file

#include <soillib/soillib.hpp>
#include <soillib/util/types.hpp>

namespace soil {

//! \todo Make sure that buffers are "re-interpretable"!

//! buffer_t<T> is a strict-typed, raw-data extent.
//! 
template<typename T>
struct buffer_t: typedbase {

  buffer_t() = default;

  buffer_t(const size_t size){ 
    this->allocate(size); 
  }
  
  ~buffer_t() override { 
    this->deallocate(); 
  }

  constexpr soil::dtype type() noexcept override { 
    return soil::typedesc<T>::type; 
  }

  // Allocator / Deallocator

  void allocate(const size_t size){
    if(this->_data != NULL)
      throw std::runtime_error("can't allocate over allocated buffer");
    if(size == 0)
      throw std::invalid_argument("size must be greater than 0");
    this->_data = std::make_shared<T[]>(size);
    this->_size = size;
  }

  void deallocate(){
    this->_data = NULL;
  }

  // Subscript Operator (Unsafe / Safe)

  T& operator[](const size_t index) noexcept {
    return this->_data[index];
  }

  T operator[](const size_t index) const noexcept {
    return this->_data[index];
  }

  // Data Inspection Member Functions

  inline size_t elem()  const { return this->_size; }
  inline size_t size()  const { return this->elem() * sizeof(T); }
  inline void* data()         { return (void*)this->_data.get(); }

private:
  std::shared_ptr<T[]> _data = NULL;  //!< Raw Data Pointer Member 
  size_t _size = 0;                   //!< Number of Data Elements
};

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

  static typedbase* make(const soil::dtype type, const size_t size) {
    return typeselect(type, [size]<typename S>() -> typedbase* {
      return new soil::buffer_t<S>(size);
    });
  }

  //! unsafe cast to strict-type
  template<typename T> inline buffer_t<T>& as() noexcept {
    return static_cast<buffer_t<T>&>(*(this->impl));
  }

  template<typename T> inline const buffer_t<T>& as() const noexcept {
    return static_cast<buffer_t<T>&>(*(this->impl));
  }

  // Inspection Operations

  void* data() {
    return typeselect(this->type(), [self=this]<typename S>(){
      return self->as<S>().data();
    });
  }
  
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

  // Lookup Operators
  // Note: These are strict typed.

  // Unsafe

  template<typename T> T& operator[](const size_t index) {
    return typeselect(this->type(), [self=this, index]<typename S>(){
      return self->as<S>().template operator[](index);
    });
  }

  template<typename T> T operator[](const size_t index) const {
    return typeselect(this->type(), [self=this, index]<typename S>(){
      return self->as<S>().template operator[](index);
    });
  }

private:
  typedbase* impl;  //!< Strict-Typed Implementation Pointer
};

} // end of namespace soil

#endif