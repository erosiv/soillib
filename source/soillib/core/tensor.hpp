#ifndef SOILLIB_TENSOR
#define SOILLIB_TENSOR

#include <soillib/core/shape.hpp>
#include <soillib/core/buffer.hpp>
#include <soillib/core/view.hpp>

namespace soil {

//! tensor_t is a strict-typed data extent with a shape
//!
template<typename T>
struct tensor_t: typedbase {

  tensor_t(){}

  tensor_t(const shape shape, const host_t host = CPU):
    _shape(shape),_buffer(shape.elem, host){}

  tensor_t(const buffer_t<T> buffer, const shape shape):
    _shape(shape),_buffer(buffer){}

  // Inspection Functions

  GPU_ENABLE shape shape()        const { return this->_shape; }
  GPU_ENABLE buffer_t<T> buffer() const { return this->_buffer; }

  GPU_ENABLE inline size_t elem() const { return this->_shape.elem; }        //!< Number of Elements
  GPU_ENABLE inline size_t size() const { return this->elem() * sizeof(T); } //!< Total Size in Bytes
  GPU_ENABLE inline T *data() { return this->_buffer.data(); }               //!< Raw Data Pointer
  GPU_ENABLE inline const T *data() const { return this->_buffer.data(); }   //!< Raw Data Pointer
  
  GPU_ENABLE inline size_t refs() const { return *this->_buffer.refs(); } //!< Internal Reference Count
  GPU_ENABLE inline host_t host() const { return this->_buffer.host(); }  //!< Current Device (CPU / GPU)
  
  // Device Changing
  void to_cpu() { this->_buffer.to_cpu(); } //!< Move Tensor Data to CPU
  void to_gpu() { this->_buffer.to_gpu(); } //!< Move Tensor Data to GPU

  //! Type Enumerator Retrieval
  constexpr soil::dtype type() noexcept override {
    return soil::typedesc<T>::type;
  }

  //! Const Subscript Operator
  GPU_ENABLE T operator[](const size_t index) const noexcept {
    return this->_buffer[index];
  }
  
  //! Non-Const Subscript Operator
  GPU_ENABLE T &operator[](const size_t index) noexcept {
    return this->_buffer[index];
  }

  template<typename S>
  GPU_ENABLE view_t<S>&& view() noexcept {
    return std::move(view_t<S>(
      reinterpret_cast<S*>(this->data()),
      this->size() / sizeof(S),
      this->host()
    ));
  };

private:
  soil::shape _shape;
  soil::buffer_t<T> _buffer;
};

//! tensor is a poylymorphic tensor_t wrapper type.
struct tensor {

  tensor() = default;
  tensor(const soil::dtype type, const soil::shape shape): impl{make(type, shape)} {}
  tensor(const soil::dtype type, const soil::shape shape, const host_t host): impl{make(type, shape, host)} {}

  //! Note that since it holds a shared pointer to a buffer_t,
  //! holding a shared pointer, if the copied or moved object
  //! is destroyed, the underlying raw memory is not deleted.

  template<typename T>
  tensor(const soil::tensor_t<T> &ten) {
    impl = std::make_shared<soil::tensor_t<T>>(ten);
  }

  template<typename T>
  tensor(soil::tensor_t<T> &&ten) {
    impl = std::make_shared<soil::tensor_t<T>>(ten);
  }

  // Construct from Buffer

  template<typename T>
  tensor(const soil::buffer_t<T>& buffer, const soil::shape &shape){
    impl = std::make_shared<soil::tensor_t<T>>(buffer, shape);
  }

  ~tensor() { this->impl = NULL; }

  //! retrieve the strict-typed type enumerator
  inline soil::dtype type() const noexcept {
    return this->impl->type();
  }

  //! unsafe cast to strict-type
  template<typename T>
  inline tensor_t<T> &as() noexcept {
    return static_cast<tensor_t<T> &>(*(this->impl));
  }

  template<typename T>
  inline const tensor_t<T> &as() const noexcept {
    return static_cast<tensor_t<T> &>(*(this->impl));
  }

  //! Const Subscript Operator
  template<typename T>
  T operator[](const size_t index) const {
    return select(this->type(), [self = this, index]<typename S>() -> T {
      if constexpr (std::same_as<S, T>) {
        return self->as<T>().operator[](index);
      } else if constexpr (std::convertible_to<S, T>) {
        return (T)self->as<S>().operator[](index);
      } else {
        throw soil::error::cast_error<S, T>();
      }
    });
  }

  //! Non-Const Subscript Operator
  template<typename T>
  T &operator[](const size_t index) {
    return this->as<T>()[index];
    //    return select(this->type(), [self = this, index]<typename S>() -> T& {
    //      if constexpr (std::same_as<S, T>) {
    //        return self->as<T>()[index];
    //      } else {
    //        throw soil::error::cast_error<S, T>();
    //      }
    //    });
  }

  // Data Inspection Operations (Type-Deducing)

  soil::shape shape() const {
    return select(this->type(), [self = this]<typename S>() {
      return self->as<S>().shape();
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

  soil::host_t host() const {
    return select(this->type(), [self = this]<typename S>() {
      return self->as<S>().host();
    });
  }

private:
  using ptr_t = std::shared_ptr<typedbase>;
  ptr_t impl; //!< Strict-Typed Implementation Base Pointer

  static ptr_t make(const soil::dtype type, const soil::shape shape, const host_t host = CPU) {
    return select(type, [shape, host]<typename S>() -> ptr_t {
      return std::make_shared<soil::tensor_t<S>>(shape, host);
    });
  }
};

}

#endif