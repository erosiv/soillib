#ifndef SOILLIB_TEXTURE
#define SOILLIB_TEXTURE

#include <cuda_runtime.h>
#include <soillib/core/buffer.hpp>
#include <soillib/core/index.hpp>

#include <iostream>

namespace soil {

//! texture is a cuda texture wrapper,
//! for interaction with raw buffer types.
//!
//! The primary use of this is to accelerate
//! lookup and reduce cache misses in kernels
//! which require lookups in const map data.
//!
template<typename T>
struct texture {

  texture(const soil::buffer_t<T> &buf, const soil::flat_t<2> &index) {

    if (buf.host() != soil::host_t::GPU) {
      throw soil::error::mismatch_host(soil::host_t::GPU, buf.host());
    }

    if constexpr (std::is_same_v<T, int>) {

      this->w = index[1]; // Index Domain Width
      this->h = index[0]; // Index Domain Height

      this->channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
      cudaMallocArray(&this->cuArray, &this->channelDesc, w, h, cudaArraySurfaceLoadStore);

      const size_t spitch = this->w * sizeof(T);
      cudaMemcpy2DToArray(this->cuArray, 0, 0, buf.data(), spitch, this->w * sizeof(T), this->h, cudaMemcpyDeviceToDevice);

      memset(&this->texDesc, 0, sizeof(this->texDesc));
      this->texDesc.normalizedCoords = 0;
      this->texDesc.readMode = cudaReadModeElementType;
      this->texDesc.addressMode[0] = cudaAddressModeClamp;
      this->texDesc.addressMode[1] = cudaAddressModeClamp;

      memset(&this->resDesc, 0, sizeof(this->resDesc));
      this->resDesc.resType = cudaResourceTypeArray;
      this->resDesc.res.array.array = this->cuArray;

      // Create texture object
      cudaCreateTextureObject(&this->texObj, &this->resDesc, &this->texDesc, NULL);
      cudaDeviceSynchronize();

    } else {

      throw std::invalid_argument("type not supported by texture");
    }
  }

  __device__ const T operator[](const soil::vec2 pos) const {
    return tex2D<T>(this->texObj, pos.y, pos.x);
  }

private:
  cudaTextureObject_t texObj = 0;

  size_t w, h;
  cudaChannelFormatDesc channelDesc;
  cudaArray_t cuArray;

  struct cudaResourceDesc resDesc;
  struct cudaTextureDesc texDesc;
};

} // namespace soil

#endif