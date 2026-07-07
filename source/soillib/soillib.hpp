#ifndef SOILLIB
#define SOILLIB

#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <type_traits>

//
// Macro Definitions
//

// Suppress Unnecessary Warnings

#ifdef HAS_CUDA
#pragma nv_diag_suppress 177
#pragma nv_diag_suppress 445
#pragma nv_diag_suppress 20011
#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20015
#endif

// Enable Host+Device Code Macro

#ifndef GPU_ENABLE
#  define GPU_ENABLE
#  ifdef HAS_CUDA
#    undef GPU_ENABLE
#    define GPU_ENABLE __host__ __device__
#  endif
#endif

// Exported Symbols from the Shared DLL Macro

#if defined(_WIN32)
#  if defined(SOIL_SHARED_BUILD)
#    define SOIL_API __declspec(dllexport)
#  else
#    define SOIL_API __declspec(dllimport)
#  endif
#else
#  if defined(SOIL_SHARED_BUILD)
#    define SOIL_API __attribute__((visibility("default")))
#  else
#    define SOIL_API
#  endif
#endif

namespace soil {

}; // namespace soil

#endif
