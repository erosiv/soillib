#ifndef SOILLIB_BUFFER_CU
#define SOILLIB_BUFFER_CU

#include <soillib/core/buffer.hpp>
#include <cuda_runtime.h>

namespace soil {

template<typename T>
void soil::buffer_t<T>::allocate(const size_t size, const host_t host) {

  if (size == 0)
    throw std::invalid_argument("size must be greater than 0");
  
  this->_size = size;

  if(host == CPU){
    this->_data = new T[size];
  }

  else if(host == GPU){
    cudaMalloc(&this->_data, this->size());
  }

  else throw std::invalid_argument("device not recognized");

  this->_host = host;
  this->_refs = new size_t(1); 

}

template<typename T>
void soil::buffer_t<T>::deallocate() {

  if(*this->_refs == 0)
  return;
		
  (*this->_refs)--;
  if(*this->_refs > 0)
    return;

  delete this->_refs;

  if(this->_data != NULL){
    if(this->_host == CPU){
      delete[] this->_data;
      this->_data = NULL;
      this->_size = 0;
      this->_host = CPU;
    }

    if(this->_host == GPU){
      cudaFree(this->_data);
      this->_data = NULL;
      this->_size = 0;
      this->_host = CPU;
    }
  }

}

template<typename T>
void soil::buffer_t<T>::to_gpu() {

  if(this->_host == GPU)
    return;

  if(this->_data == NULL)
    return;

  if(this->_size == 0)
    return;
  
  T* _data;
  size_t _size = this->_size;

  cudaMalloc(&_data, this->size());
  cudaMemcpy(_data, this->data(), this->size(), cudaMemcpyHostToDevice);
  
  this->deallocate();
  this->_data = _data;
  this->_refs = new size_t(1);
  this->_size = _size;
  this->_host = GPU;

}

template<typename T>
void soil::buffer_t<T>::to_cpu() {

  if(this->_host == CPU)
    return;

  if(this->_data == NULL)
    return;

  if(this->_size == 0)
    return;

  size_t _size = this->_size;
  T* _data = new T[_size];
  cudaMemcpy(_data, this->data(), this->size(), cudaMemcpyDeviceToHost);
  this->deallocate();

  this->_data = _data;
  this->_refs = new size_t(1);
  this->_size = _size;
  this->_host = CPU;

}

template struct soil::buffer_t<int>;
template struct soil::buffer_t<float>;
template struct soil::buffer_t<double>;
template struct soil::buffer_t<vec2>;
template struct soil::buffer_t<vec3>;
template struct soil::buffer_t<ivec2>;
template struct soil::buffer_t<ivec3>;

} // end of namespace soil

#endif