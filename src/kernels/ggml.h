#pragma once

// Standalone shim replacing ggml's public header.
//
// ggml-quants.c / ggml-quants.h in this directory are verbatim copies of the
// upstream ggml sources (see ggml/src/). Instead of editing them, this shim
// provides the minimal subset of ggml.h they compile against: the type enum,
// fp16/bf16 typedefs, assert helpers and the handful of type-trait functions,
// which are implemented locally in qz-traits.c / qz-init.cpp.
//
// To re-sync the kernels with a newer ggml, just re-copy ggml-quants.c,
// ggml-quants.h and ggml-common.h - no other changes should be needed.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GGML_API

#ifdef __GNUC__
#  define GGML_RESTRICT __restrict__
#elif defined(_MSC_VER)
#  define GGML_RESTRICT __restrict
#else
#  define GGML_RESTRICT restrict
#endif

#define GGML_UNUSED(x) (void)(x)

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void ggml_abort_impl(const char * file, int line, const char * fmt, ...);

#ifdef __cplusplus
}
#endif

#define GGML_ABORT(...) ggml_abort_impl(__FILE__, __LINE__, __VA_ARGS__)
#define GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            GGML_ABORT("GGML_ASSERT(%s) failed", #x); \
        } \
    } while (0)

#if defined(__GNUC__)
#  define GGML_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#  define GGML_UNREACHABLE() __assume(false)
#else
#  define GGML_UNREACHABLE() ((void) 0)
#endif

#define GGML_MAX_DIMS 4

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t ggml_fp16_t;

typedef struct {
    uint16_t bits;
} ggml_bf16_t;

// NOTE: mirrors the upstream ggml enum - the numeric values are part of the
// GGUF file format and must never change.
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
    GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
    GGML_TYPE_Q1_0    = 41,
    GGML_TYPE_COUNT   = 42,
};

// type traits (implemented in qz-traits.c)
GGML_API const char * ggml_type_name(enum ggml_type type);
GGML_API int64_t      ggml_blck_size(enum ggml_type type);
GGML_API size_t       ggml_type_size(enum ggml_type type);                    // size of one block
GGML_API size_t       ggml_row_size(enum ggml_type type, int64_t ne);         // ne must be a multiple of the block size
GGML_API bool         ggml_is_quantized(enum ggml_type type);

// f16/bf16 row conversions (implemented in qz-traits.c)
GGML_API float ggml_fp16_to_fp32(ggml_fp16_t x);
GGML_API ggml_fp16_t ggml_fp32_to_fp16(float x);
GGML_API void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n);
GGML_API void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n);
GGML_API void ggml_bf16_to_fp32_row(const ggml_bf16_t * x, float * y, int64_t n);
GGML_API void ggml_fp32_to_bf16_row_ref(const float * x, ggml_bf16_t * y, int64_t n);

// quantization entry points
GGML_API void   ggml_quantize_init(enum ggml_type type);   // thread-safe, implemented in qz-init.cpp
GGML_API void   ggml_quantize_free(void);
GGML_API bool   ggml_quantize_requires_imatrix(enum ggml_type type);
GGML_API size_t ggml_quantize_chunk(
        enum ggml_type   type,
           const float * src,
                  void * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           const float * imatrix);

// defined in ggml-quants.c
GGML_API bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes);

// dequantize one contiguous run of `k` elements to f32; returns false if the
// source type has no to_float implementation (implemented in qz-traits.c)
GGML_API bool qz_dequantize(enum ggml_type type, const void * src, float * dst, int64_t k);

// true if `type` is one of the types ggml_quantize_chunk() can produce
GGML_API bool qz_is_quantize_target(enum ggml_type type);

#ifdef __cplusplus
}
#endif
