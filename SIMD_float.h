#if defined(__AVX512F__)
#include "SIMD_float_512.h" // Header for AVX-512 intrinsics
#elif defined(__AVX2__)
#include "SIMD_float_256.h" // Header for AVX2 intrinsics
#elif defined(__AVX__)
#include "SIMD_float_256.h" // Header for AVX intrinsics
#elif defined(_M_IX86_FP) && _M_IX86_FP == 2
#include "SIMD_float_128.h" // Header for SSE2 intrinsics
#elif defined(_M_IX86_FP) && _M_IX86_FP == 1
#include "SIMD_float_128.h" // Header for SSE intrinsics
#elif defined(_M_X64)
#include "SIMD_float_128.h" // SSE2 support is implied for x64
#else
#error "No SIMD support detected. Ensure your compiler supports at least SSE."
#endif