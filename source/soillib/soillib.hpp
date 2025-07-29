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

#if defined(SHARED_BUILD)
#  define EXPORT_SHARED __declspec(dllexport)
#else
#  define EXPORT_SHARED __declspec(dllexport)
#endif

namespace soil {

}; // namespace soil

#endif
