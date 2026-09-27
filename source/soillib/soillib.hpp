#pragma once

//
// soillib umbrella header.
//

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>

//
// Diagnostic suppression
//

#if defined(HAS_CUDA) && defined(__CUDACC__)
#pragma nv_diag_suppress 177 // variable declared but never referenced (unused template parameters)
#pragma nv_diag_suppress 445 // constant not used in declaring parameter types (used only for template dispatch)
#pragma nv_diag_suppress 20011
#pragma nv_diag_suppress 20012 // __host__ ignored on defaulted special member (fires from glm's mat headers)
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#endif

//
// Host+device function macro
//

#ifndef GPU_ENABLE
#define GPU_ENABLE
#ifdef HAS_CUDA
#undef GPU_ENABLE
#define GPU_ENABLE __host__ __device__
#endif
#endif

//
// CUDA error checking
//

#include <silt/core/error.hpp>

//
// Version
//
// SOIL_VERSION_* are defined by the build from the root VERSION file, the
// single source of truth (see source/CMakeLists.txt). SOIL_VERSION_NUM is the
// comparable form, for downstream guarding:
//
//   #if SOIL_VERSION_NUM >= 10200
//

namespace soil {

} // namespace soil
