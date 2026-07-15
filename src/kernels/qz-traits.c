// Local implementation of the small slice of ggml.c that the vendored
// ggml-quants.c (and the quantizer itself) depend on: the type-traits table,
// f16/bf16 row conversions and the quantize/dequantize dispatchers.

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include <stdarg.h>
#include <string.h>

void ggml_abort_impl(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);
    fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

struct qz_type_traits {
    const char * type_name;
    int64_t      blck_size;
    size_t       type_size;
    bool         is_quantized;
};

static const struct qz_type_traits type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32]     = { "f32",     1,      sizeof(float),          false },
    [GGML_TYPE_F16]     = { "f16",     1,      sizeof(ggml_fp16_t),    false },
    [GGML_TYPE_Q4_0]    = { "q4_0",    QK4_0,  sizeof(block_q4_0),     true  },
    [GGML_TYPE_Q4_1]    = { "q4_1",    QK4_1,  sizeof(block_q4_1),     true  },
    [GGML_TYPE_Q5_0]    = { "q5_0",    QK5_0,  sizeof(block_q5_0),     true  },
    [GGML_TYPE_Q5_1]    = { "q5_1",    QK5_1,  sizeof(block_q5_1),     true  },
    [GGML_TYPE_Q8_0]    = { "q8_0",    QK8_0,  sizeof(block_q8_0),     true  },
    [GGML_TYPE_Q8_1]    = { "q8_1",    QK8_1,  sizeof(block_q8_1),     true  },
    [GGML_TYPE_Q2_K]    = { "q2_K",    QK_K,   sizeof(block_q2_K),     true  },
    [GGML_TYPE_Q3_K]    = { "q3_K",    QK_K,   sizeof(block_q3_K),     true  },
    [GGML_TYPE_Q4_K]    = { "q4_K",    QK_K,   sizeof(block_q4_K),     true  },
    [GGML_TYPE_Q5_K]    = { "q5_K",    QK_K,   sizeof(block_q5_K),     true  },
    [GGML_TYPE_Q6_K]    = { "q6_K",    QK_K,   sizeof(block_q6_K),     true  },
    [GGML_TYPE_Q8_K]    = { "q8_K",    QK_K,   sizeof(block_q8_K),     true  },
    [GGML_TYPE_IQ2_XXS] = { "iq2_xxs", QK_K,   sizeof(block_iq2_xxs),  true  },
    [GGML_TYPE_IQ2_XS]  = { "iq2_xs",  QK_K,   sizeof(block_iq2_xs),   true  },
    [GGML_TYPE_IQ3_XXS] = { "iq3_xxs", QK_K,   sizeof(block_iq3_xxs),  true  },
    [GGML_TYPE_IQ1_S]   = { "iq1_s",   QK_K,   sizeof(block_iq1_s),    true  },
    [GGML_TYPE_IQ4_NL]  = { "iq4_nl",  QK4_NL, sizeof(block_iq4_nl),   true  },
    [GGML_TYPE_IQ3_S]   = { "iq3_s",   QK_K,   sizeof(block_iq3_s),    true  },
    [GGML_TYPE_IQ2_S]   = { "iq2_s",   QK_K,   sizeof(block_iq2_s),    true  },
    [GGML_TYPE_IQ4_XS]  = { "iq4_xs",  QK_K,   sizeof(block_iq4_xs),   true  },
    [GGML_TYPE_I8]      = { "i8",      1,      sizeof(int8_t),         false },
    [GGML_TYPE_I16]     = { "i16",     1,      sizeof(int16_t),        false },
    [GGML_TYPE_I32]     = { "i32",     1,      sizeof(int32_t),        false },
    [GGML_TYPE_I64]     = { "i64",     1,      sizeof(int64_t),        false },
    [GGML_TYPE_F64]     = { "f64",     1,      sizeof(double),         false },
    [GGML_TYPE_IQ1_M]   = { "iq1_m",   QK_K,   sizeof(block_iq1_m),    true  },
    [GGML_TYPE_BF16]    = { "bf16",    1,      sizeof(ggml_bf16_t),    false },
    [GGML_TYPE_TQ1_0]   = { "tq1_0",   QK_K,   sizeof(block_tq1_0),    true  },
    [GGML_TYPE_TQ2_0]   = { "tq2_0",   QK_K,   sizeof(block_tq2_0),    true  },
    [GGML_TYPE_MXFP4]   = { "mxfp4",   QK_MXFP4,  sizeof(block_mxfp4), true  },
    [GGML_TYPE_NVFP4]   = { "nvfp4",   QK_NVFP4,  sizeof(block_nvfp4), true  },
    [GGML_TYPE_Q1_0]    = { "q1_0",    QK1_0,  sizeof(block_q1_0),     true  },
};

const char * ggml_type_name(enum ggml_type type) {
    if ((int) type < 0 || type >= GGML_TYPE_COUNT) {
        return NULL;
    }
    return type_traits[type].type_name; // NULL for removed/unsupported ids
}

int64_t ggml_blck_size(enum ggml_type type) {
    return type_traits[type].blck_size;
}

size_t ggml_type_size(enum ggml_type type) {
    return type_traits[type].type_size;
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    GGML_ASSERT(ne % ggml_blck_size(type) == 0);
    return ggml_type_size(type) * ne / ggml_blck_size(type);
}

bool ggml_is_quantized(enum ggml_type type) {
    return type_traits[type].is_quantized;
}

float ggml_fp16_to_fp32(ggml_fp16_t x) {
    return GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(float x) {
    return GGML_FP32_TO_FP16(x);
}

void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

void ggml_bf16_to_fp32_row(const ggml_bf16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

void ggml_fp32_to_bf16_row_ref(const float * x, ggml_bf16_t * y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

bool ggml_quantize_requires_imatrix(enum ggml_type type) {
    return
        type == GGML_TYPE_IQ2_XXS ||
        type == GGML_TYPE_IQ2_XS  ||
        type == GGML_TYPE_IQ1_S;
}

bool qz_is_quantize_target(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

size_t ggml_quantize_chunk(
        enum ggml_type   type,
           const float * src,
                  void * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           const float * imatrix) {
    const int64_t n = nrows * n_per_row;

    if (ggml_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != NULL);
    }

    GGML_ASSERT(start % ggml_blck_size(type) == 0);
    GGML_ASSERT(start % n_per_row == 0);

    ggml_quantize_init(type); // this is noop if already initialized

    const size_t start_row = start / n_per_row;
    const size_t row_size  = ggml_row_size(type, n_per_row);

    size_t result = 0;

    switch (type) {
        case GGML_TYPE_Q1_0:    result = quantize_q1_0   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_Q4_0:    result = quantize_q4_0   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_Q4_1:    result = quantize_q4_1   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_Q5_0:    result = quantize_q5_0   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_Q5_1:    result = quantize_q5_1   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_Q8_0:    result = quantize_q8_0   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_MXFP4:   result = quantize_mxfp4  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_NVFP4:   result = quantize_nvfp4  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_Q2_K:    result = quantize_q2_K   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_Q3_K:    result = quantize_q3_K   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_Q4_K:    result = quantize_q4_K   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_Q5_K:    result = quantize_q5_K   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_Q6_K:    result = quantize_q6_K   (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQ1_0:   result = quantize_tq1_0  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_IQ3_S:   result = quantize_iq3_s  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_IQ2_S:   result = quantize_iq2_s  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_IQ1_S:   result = quantize_iq1_s  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_IQ1_M:   result = quantize_iq1_m  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case GGML_TYPE_F16:
            {
                size_t elemsize = sizeof(ggml_fp16_t);
                ggml_fp32_to_fp16_row(src + start, (ggml_fp16_t *)dst + start, n);
                result = n * elemsize;
            } break;
        case GGML_TYPE_BF16:
            {
                size_t elemsize = sizeof(ggml_bf16_t);
                ggml_fp32_to_bf16_row_ref(src + start, (ggml_bf16_t *)dst + start, n);
                result = n * elemsize;
            } break;
        case GGML_TYPE_F32:
            {
                size_t elemsize = sizeof(float);
                result = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, result);
            } break;
        default:
            GGML_ABORT("invalid quantization target type %d", (int) type);
    }

    GGML_ASSERT(result == nrows * row_size);

    return result;
}

bool qz_dequantize(enum ggml_type type, const void * src, float * dst, int64_t k) {
    switch (type) {
        case GGML_TYPE_F32:     memcpy(dst, src, (size_t) k * sizeof(float));                     break;
        case GGML_TYPE_F16:     ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, k);         break;
        case GGML_TYPE_BF16:    ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, k);         break;
        case GGML_TYPE_Q1_0:    dequantize_row_q1_0   ((const block_q1_0    *) src, dst, k);      break;
        case GGML_TYPE_Q4_0:    dequantize_row_q4_0   ((const block_q4_0    *) src, dst, k);      break;
        case GGML_TYPE_Q4_1:    dequantize_row_q4_1   ((const block_q4_1    *) src, dst, k);      break;
        case GGML_TYPE_Q5_0:    dequantize_row_q5_0   ((const block_q5_0    *) src, dst, k);      break;
        case GGML_TYPE_Q5_1:    dequantize_row_q5_1   ((const block_q5_1    *) src, dst, k);      break;
        case GGML_TYPE_Q8_0:    dequantize_row_q8_0   ((const block_q8_0    *) src, dst, k);      break;
        case GGML_TYPE_MXFP4:   dequantize_row_mxfp4  ((const block_mxfp4   *) src, dst, k);      break;
        case GGML_TYPE_NVFP4:   dequantize_row_nvfp4  ((const block_nvfp4   *) src, dst, k);      break;
        case GGML_TYPE_Q2_K:    dequantize_row_q2_K   ((const block_q2_K    *) src, dst, k);      break;
        case GGML_TYPE_Q3_K:    dequantize_row_q3_K   ((const block_q3_K    *) src, dst, k);      break;
        case GGML_TYPE_Q4_K:    dequantize_row_q4_K   ((const block_q4_K    *) src, dst, k);      break;
        case GGML_TYPE_Q5_K:    dequantize_row_q5_K   ((const block_q5_K    *) src, dst, k);      break;
        case GGML_TYPE_Q6_K:    dequantize_row_q6_K   ((const block_q6_K    *) src, dst, k);      break;
        case GGML_TYPE_TQ1_0:   dequantize_row_tq1_0  ((const block_tq1_0   *) src, dst, k);      break;
        case GGML_TYPE_TQ2_0:   dequantize_row_tq2_0  ((const block_tq2_0   *) src, dst, k);      break;
        case GGML_TYPE_IQ2_XXS: dequantize_row_iq2_xxs((const block_iq2_xxs *) src, dst, k);      break;
        case GGML_TYPE_IQ2_XS:  dequantize_row_iq2_xs ((const block_iq2_xs  *) src, dst, k);      break;
        case GGML_TYPE_IQ2_S:   dequantize_row_iq2_s  ((const block_iq2_s   *) src, dst, k);      break;
        case GGML_TYPE_IQ3_XXS: dequantize_row_iq3_xxs((const block_iq3_xxs *) src, dst, k);      break;
        case GGML_TYPE_IQ3_S:   dequantize_row_iq3_s  ((const block_iq3_s   *) src, dst, k);      break;
        case GGML_TYPE_IQ1_S:   dequantize_row_iq1_s  ((const block_iq1_s   *) src, dst, k);      break;
        case GGML_TYPE_IQ1_M:   dequantize_row_iq1_m  ((const block_iq1_m   *) src, dst, k);      break;
        case GGML_TYPE_IQ4_NL:  dequantize_row_iq4_nl ((const block_iq4_nl  *) src, dst, k);      break;
        case GGML_TYPE_IQ4_XS:  dequantize_row_iq4_xs ((const block_iq4_xs  *) src, dst, k);      break;
        default:
            return false; // ints, f64, q8_1/q8_K intermediates, removed types
    }
    return true;
}
