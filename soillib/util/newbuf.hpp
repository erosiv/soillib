#ifndef SOILLIB_UTIL_BUF_NEW
#define SOILLIB_UTIL_BUF_NEW

#include <soillib/soillib.hpp>

namespace soil {
namespace nn {

//! buf<T> is a strict-typed, owning raw-data
//! extent with a built-in iterator.
//!
//! buf<T> is intended as a low-level base
//! structure for memory-pooled applications.
//!

struct type {

  // we can probably wrap most of this behavior
  // in some kind of typeinfo structure.
  // this should probably be as constexpr as possible,
  // with static pre-defined types, so that it is easy
  // to construct different polymorphic dynamic containers.

};

//! Buffer Base-Type for Polymorphic Inheritance
//! Strict-Typed Buffer Common Interface
struct buf_base {

  virtual ~buf_base() = default;

  virtual void allocate(const size_t size)  = 0;  //!< Allocate Memory Buffer
  virtual void deallocate()                 = 0;  //!< De-Allocate Memory Buffer

  virtual void*  data() = 0;        //!< Retrieve Raw Data Pointer
  virtual size_t size() const = 0;  //!< Retrieve Size of Buffer in Bytes
  virtual size_t elem() const = 0;  //!< Retreive Number of Typed Elements
};

//! buf_t is the strict-typed buffer implementation.
//! Inheriting from buf_base, it satisfies the interface.
//!< Strict-Typed Buffer Implementation
template<typename T> 
struct buf_t: public buf_base {

  buf_t(){} // default zero-constructor
  
  buf_t(const size_t size){
    this->allocate(size);
  }

  ~buf_t(){
    this->deallocate();
  }

  void allocate(const size_t size){
    if(this->_data == NULL){
      this->_data = new T[size];
      this->_size = size;
    }
  }

  void deallocate(){
    if(this->_data != NULL){
      delete[] this->_data;
      this->_size = 0;
    }
  }

  template<typename F>
  F at(const size_t ind){
    return (F)*(this->data + ind);
  }

  // Member Function Implementations

  void* data()        { return (void*)this->_data; }
  size_t size() const { return sizeof(T) * this->_size; }
  size_t elem() const { return this->_size; }

private:

  T* _data = NULL;  //!< Raw Data Pointer Member
  size_t _size = 0; //!< Data Size in Bytes Member

public:

  struct iterator {

    iterator() noexcept : start(NULL),ind(0){};
    iterator(T* t, size_t ind) noexcept : start(t),ind(ind){};

    // Base Operators

    T& operator*() noexcept {
      return *(this->start+ind);
    };

    const iterator& operator++() noexcept {
      ind++;
      return *this;
    };

    const bool operator==(const iterator& other) const noexcept {
      return this->start == other.start && this->ind == other.ind;
    };

    const bool operator!=(const iterator& other) const noexcept {
      return !(*this == other);
    };

    private:
      T* start;
      size_t ind;
  };

  const iterator begin() const noexcept { return iterator(this->_data, 0); }
  const iterator end()   const noexcept { return iterator(this->_data, _size); }
};

/*
*/
//! buf is a polymorphic dynamically typed buffer container
//!
//! buf contains a strict-typed, memory-safe buffer struct,
//! and provides and interface so that underlying type is
//! runtime constructable for dynamic memory typing.
//!
struct buf {

  buf(const char* type):
    _buf(make(type)){}
  
  ~buf(){
    if(this->_buf != NULL){
      delete this->_buf;
    }
  }

  //! Underlying Buffer-Type Factory Function
  static buf_base* make(const char* type){
    //! move this to the type structure
    if(type == "int")     return new buf_t<int>();
    if(type == "float")   return new buf_t<float>();
    if(type == "double")  return new buf_t<double>();
    return NULL;
  }

  //! Strict-Typed Buffer Implementation Retrieval
  template<typename T> buf_t<T> as(){
    return dynamic_cast<buf_t<T>>(_buf);
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

  void* data() { 
    return this->_buf->data(); 
  }

  buf_base* _buf = NULL;
};

}
}

#endif