#ifndef SOILLIB_TENSOR
#define SOILLIB_TENSOR

#include <soillib/soillib.hpp>
#include <soillib/core/error.hpp>
#include <soillib/core/shape.hpp>
#include <soillib/core/view.hpp>
#include <cuda_runtime.h>

namespace soil {

//! tensor_t<T> is a strict-typed, owning raw-data extent.
//!
//! tensor_t<T> is reference counting with correct move semantics,
//! making copying and moving tensor_t<T> values cheap and easy.
//!
//! tensor_t<T> data can be on the CPU or on the GPU.
//!
template<typename T>
struct tensor_t: typedbase {

  //
  // Construction and Assignment
  //

  //! Default Empty Constructor
  tensor_t() {
    this->_shape = shape();
    this->_data = NULL;
    this->_refs = new size_t(0);
    this->_host = CPU;
  }

  //! Allocating Constructor
  tensor_t(const shape shape, const host_t host = CPU) {
    this->allocate(shape, host);
  }

  //! Non-Allocating Constructor
  tensor_t(T* data, const shape shape, const host_t host = CPU){
    this->_data = data;
    this->_refs = NULL;
    this->_shape = shape;
    this->_host = host;
  }

  ~tensor_t() { this->deallocate(); }

  //! Copy Constructor (Reference Increment)
  tensor_t(const tensor_t<T> &other) {
    this->_data = other._data;
    this->_refs = other._refs;
    this->_shape = other._shape;
    this->_host = other._host;
    if (this->_data != NULL) {
      if (this->_refs != NULL){
        ++(*this->_refs);
      }
    }
  }

  //! Copy Assignment Operator (Reference Increment)
  tensor_t &operator=(const tensor_t<T> &other) {
    this->deallocate();
    this->_data = other._data;
    this->_refs = other._refs;
    this->_shape = other._shape;
    this->_host = other._host;
    if (this->_data != NULL) {
      if (this->_refs != NULL) {
        ++(*this->_refs);
      }
    }
    return *this;
  }

  //! Move Constructor (Reference Steal)
  tensor_t(tensor_t<T> &&other) {
    this->_data = other._data;
    this->_refs = other._refs;
    this->_shape = other._shape;
    this->_host = other._host;
    other._data = NULL;
    other._refs = NULL;
  }

  //! Move Assginmment Operator (Reference Steal)
  tensor_t &operator=(tensor_t<T> &&other) {
    this->deallocate();
    this->_data = other._data;
    this->_refs = other._refs;
    this->_shape = other._shape;
    this->_host = other._host;
    other._data = NULL;
    other._refs = NULL;
    return *this;
  }

  //
  // Data Inspection
  //

  GPU_ENABLE inline shape shape()   const { return this->_shape; }
  GPU_ENABLE inline size_t elem()   const { return this->_shape.elem; }         //!< Number of Elements
  GPU_ENABLE inline size_t size()   const { return this->elem() * sizeof(T); }  //!< Total Size in Bytes
  GPU_ENABLE inline size_t refs()   const { return *this->_refs; }              //!< Reference Count
  GPU_ENABLE inline host_t host()   const { return this->_host; }               //!< Current Device (CPU / GPU)
  GPU_ENABLE inline const T *data() const { return this->_data; }               //!< Raw Data Pointer (Const)
  GPU_ENABLE inline T *data()             { return this->_data; }               //!< Raw Data Pointer (Mutable)
  
  //! Type Enumerator Retrieval
  constexpr soil::dtype type() noexcept {
    return soil::typedesc<T>::type;
  }

  //! Const Subscript Operator (Flat)
  GPU_ENABLE T operator[](const size_t index) const noexcept {
    return this->_data[index];
  }

  //! Non-Const Subscript Operator (Float)
  GPU_ENABLE T &operator[](const size_t index) noexcept {
    return this->_data[index];
  }

  template<typename S>
  GPU_ENABLE view_t<S> view() noexcept {
    return view_t<S>(
      reinterpret_cast<S*>(this->data()),
      this->size() / sizeof(S),
      this->host()
    );
  };

  template<typename S>
  GPU_ENABLE const_view_t<S> view() const noexcept {
    return const_view_t<S>(
      reinterpret_cast<const S*>(this->data()),
      this->size() / sizeof(S),
      this->host()
    );
  };

  void to_cpu(); //!< In-Place Copy Data to the CPU
  void to_gpu(); //!< In-Place Copy Data to the GPU (if available)

  size_t *_refs = NULL; //!< Pointer to Reference Count
private:

  //! Device-Aware Allocation and De-Allocation
  void allocate(const soil::shape shape, const host_t host = CPU);
  void deallocate();

  soil::shape _shape;   //!< Shape of Data
  host_t _host = CPU;   //!< Currently Active Device
  T *_data = NULL;      //!< Raw Data Pointer (Device Agnostic)
};

//
// Member Function Implementations
//

template<typename T>
void soil::tensor_t<T>::allocate(const soil::shape shape, const host_t host) {

  if (shape.elem == 0)
    throw std::invalid_argument("size must be greater than 0");
  this->_shape = shape;

  if (host == CPU) {
    this->_data = new T[shape.elem];
  } else if (host == GPU) {
    cudaMalloc(&this->_data, this->size());
  } else {
    throw std::invalid_argument("device not recognized");
  }

  this->_host = host;
  this->_refs = new size_t(1);

}

template<typename T>
void soil::tensor_t<T>::deallocate() {

  if (this->_refs == NULL)
    return;

  if (*this->_refs == 0)
    return;

  (*this->_refs)--;
  if (*this->_refs > 0)
    return;

  delete this->_refs;
  this->_refs = NULL;

  if (this->_data != NULL) {
    if (this->_host == CPU) {
      delete[] this->_data;
      this->_data = NULL;
      this->_host = CPU;
    }

    if (this->_host == GPU) {

      cudaFree(this->_data);
      this->_data = NULL;
      this->_host = CPU;
    }
  }
}

template<typename T>
void soil::tensor_t<T>::to_gpu() {

  if (this->_host == GPU)
    return;

  if (this->_data == NULL)
    return;

  if (this->elem() == 0)
    return;

  T *_data;

  cudaMalloc(&_data, this->size());
  cudaMemcpy(_data, this->data(), this->size(), cudaMemcpyHostToDevice);

  this->deallocate();
  this->_data = _data;
  this->_refs = new size_t(1);
  this->_host = GPU;
}

template<typename T>
void soil::tensor_t<T>::to_cpu() {

  if (this->_host == CPU)
    return;

  if (this->_data == NULL)
    return;

  if (this->elem() == 0)
    return;

  T *_data = new T[this->elem()];
  cudaMemcpy(_data, this->data(), this->size(), cudaMemcpyDeviceToHost);

  this->deallocate();
  this->_data = _data;
  this->_refs = new size_t(1);
  this->_host = CPU;
}

//! tensor is a tag-poylymorphic tensor_t wrapper type.
//!
//! This type stores a pointer to a tensor_t<T>, which it
//! can cast to the appropriate type when required.
//!
//! Because of the dll exported type, tensor cannot use
//! std::shared pointer, which is why we do the funny
//! business with the move and copy constructors:
struct EXPORT_SHARED tensor {

  tensor() = default;
  tensor(const soil::dtype type, const soil::shape shape): impl{make(type, shape)} {}
  tensor(const soil::dtype type, const soil::shape shape, const host_t host): impl{make(type, shape, host)} {}

  //! Polymorphic Tensor Copy Constructor
  tensor(const soil::tensor& rhs){
    this->impl = rhs.clone();
  }

  //! Polymorphic Tensor Move Constructor
  tensor(soil::tensor&& rhs){
    this->impl = rhs.clone();
  }

  //! Strict-Typed Tensor Copy Constructor
  template<typename T>
  tensor(const soil::tensor_t<T> &ten) {
    this->impl = new soil::tensor_t<T>(ten);
  }

  //! Strict-Typed Tensor Move Constructor
  template<typename T>
  tensor(soil::tensor_t<T> &&ten) {
    this->impl = new soil::tensor_t<T>(ten);
  }

  ~tensor() { this->clear(); }

  //! Copy Assignment Operator
  tensor& operator=(const soil::tensor& rhs) {
    this->clear();
    this->impl = rhs.clone();
    return *this;
  }

  //! Move Assignment Operator
  tensor& operator=(soil::tensor &&rhs) {
    this->clear();
    this->impl = rhs.clone();
    return *this;
  }

  //! Polymorphic Strict-Type Cast (Const)
  template<typename T>
  inline const tensor_t<T> &as() const noexcept {
    return static_cast<tensor_t<T> &>(*(this->impl));
  }

  //! Polymorphic Strict-Type Cast (Mutable)
  template<typename T>
  inline tensor_t<T> &as() noexcept {
    return static_cast<tensor_t<T> &>(*(this->impl));
  }

  //
  // Data Inspection Operations (Type-Deducing)
  //

  inline soil::dtype type() const noexcept {
    return this->impl->type();
  }

  soil::shape shape() const {
    return select(this->type(), [self = this]<typename S>() {
      return self->as<S>().shape();
    });
  }

  soil::host_t host() const {
    return select(this->type(), [self = this]<typename S>() {
      return self->as<S>().host();
    });
  }

  size_t elem() const {
    return select(this->type(), [self = this]<typename S>() {
      return self->as<S>().elem();
    });
  }

  size_t size() const {
    return select(this->type(), [self = this]<typename S>() {
      return self->as<S>().size();
    });
  }

  void *data() {
    return select(this->type(), [self = this]<typename S>() {
      return (void *)self->as<S>().data();
    });
  }

private:

  void clear() {
    if(this->impl != NULL)
      delete this->impl;
    this->impl = NULL;
  }

  //! Make a new Strict-Typed Tensor 
  static typedbase* make(const soil::dtype type, const soil::shape shape, const host_t host = CPU) {
    return select(type, [shape, host]<typename S>() -> typedbase* {
      return new soil::tensor_t<S>(shape, host);
    });
  }

  //! Clone the implementation pointer with new
  typedbase* clone() const {
    if(this->impl == NULL) 
      return NULL;
    return select(this->type(), [self = this]<typename S>() -> typedbase* {
      return new soil::tensor_t<S>(self->as<S>());
    });
  }

  typedbase* impl = NULL; //!< Polymorphic Implementation Pointer
};

}

#endif