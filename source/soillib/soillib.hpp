#ifndef SOILLIB
#define SOILLIB

#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <type_traits>

#ifndef GPU_ENABLE
#define GPU_ENABLE
#ifdef HAS_CUDA
#undef GPU_ENABLE
#define GPU_ENABLE __host__ __device__
#endif
#endif

#ifdef HAS_CUDA
#pragma nv_diag_suppress 177
#pragma nv_diag_suppress 445
#pragma nv_diag_suppress 20011
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#endif

namespace soil {

}; // namespace soil

#endif
