#pragma once
// Shim for the vendored ggml-quants.c: the scalar reference kernels need
// nothing from upstream ggml-cpu-impl.h, but ggml_validate_row_data has
// vectorized paths (NEON on ARM, AVX2 on x86) that need the SIMD headers
// upstream pulls in here.

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || \
    defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif
