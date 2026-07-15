// Apple Metal accelerator backend: fast f32 -> quant kernels for the simple
// block formats (ported from ggml's Metal cpy kernels) plus f16/bf16.
// K-quants and iq-quants have no GPU kernels and fall back to the CPU path.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "device_metal.h"

#include <algorithm>
#include <cstring>

namespace qz {
namespace {

// block layouts must match ggml-common.h bit-for-bit (see device_gpu.cu for
// the reference C++ definitions; MSL half is 2 bytes, so layouts line up)
constexpr const char * QZ_METAL_SOURCE = R"MSL(
#include <metal_stdlib>
using namespace metal;

#define QK 32 // block size of all supported quant formats

typedef struct { half d;         uint8_t qs[QK/2]; } blk_q4_0;
typedef struct { half d; half m; uint8_t qs[QK/2]; } blk_q4_1;
typedef struct { half d;         uint8_t qh[4]; uint8_t qs[QK/2]; } blk_q5_0;
typedef struct { half d; half m; uint8_t qh[4]; uint8_t qs[QK/2]; } blk_q5_1;
typedef struct { half d;         int8_t qs[QK]; } blk_q8_0;
typedef struct { half d;         uint8_t qs[QK/2]; } blk_iq4_nl;

constant int8_t k_iq4nl_values[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                                          1,   13,  25,  38,  53,  69,  89, 113 };

kernel void quantize_f16(device const float * x [[buffer(0)]],
                         device       half  * y [[buffer(1)]],
                         constant   int64_t & n [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < (uint64_t) n) {
        y[i] = (half) x[i];
    }
}

kernel void quantize_bf16(device const float    * x [[buffer(0)]],
                          device       uint16_t * y [[buffer(1)]],
                          constant   int64_t    & n [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < (uint64_t) n) {
        uint u = as_type<uint>(x[i]);
        if ((u & 0x7fffffff) > 0x7f800000) { // nan
            y[i] = (uint16_t) ((u >> 16) | 64); // force to quiet
        } else {
            y[i] = (uint16_t) ((u + (0x7fff + ((u >> 16) & 1))) >> 16);
        }
    }
}

kernel void quantize_q4_0(device const float    * x  [[buffer(0)]],
                          device       blk_q4_0 * yy [[buffer(1)]],
                          constant   int64_t    & nb [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= (uint64_t) nb) return;
    device const float * xb = x + (uint64_t) i * QK;
    device blk_q4_0 * y = yy + i;

    float amax = 0.0f;
    float vmax = 0.0f;
    for (int j = 0; j < QK; ++j) {
        const float v = xb[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = (half) d;

    for (int j = 0; j < QK/2; ++j) {
        const float x0 = xb[0    + j]*id;
        const float x1 = xb[QK/2 + j]*id;
        const uint8_t xi0 = min(15, (int)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int)(x1 + 8.5f));
        y->qs[j] = xi0 | (xi1 << 4);
    }
}

kernel void quantize_q4_1(device const float    * x  [[buffer(0)]],
                          device       blk_q4_1 * yy [[buffer(1)]],
                          constant   int64_t    & nb [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= (uint64_t) nb) return;
    device const float * xb = x + (uint64_t) i * QK;
    device blk_q4_1 * y = yy + i;

    float vmin =  FLT_MAX;
    float vmax = -FLT_MAX;
    for (int j = 0; j < QK; ++j) {
        const float v = xb[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = (half) d;
    y->m = (half) vmin;

    for (int j = 0; j < QK/2; ++j) {
        const float x0 = (xb[0    + j] - vmin)*id;
        const float x1 = (xb[QK/2 + j] - vmin)*id;
        const uint8_t xi0 = min(15, (int)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int)(x1 + 0.5f));
        y->qs[j] = xi0 | (xi1 << 4);
    }
}

kernel void quantize_q5_0(device const float    * x  [[buffer(0)]],
                          device       blk_q5_0 * yy [[buffer(1)]],
                          constant   int64_t    & nb [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= (uint64_t) nb) return;
    device const float * xb = x + (uint64_t) i * QK;
    device blk_q5_0 * y = yy + i;

    float amax = 0.0f;
    float vmax = 0.0f;
    for (int j = 0; j < QK; ++j) {
        const float v = xb[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = (half) d;

    uint32_t qh = 0;
    for (int j = 0; j < QK/2; ++j) {
        const float x0 = xb[0    + j]*id;
        const float x1 = xb[QK/2 + j]*id;
        const uint8_t xi0 = min(31, (int)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int)(x1 + 16.5f));
        y->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK/2);
    }
    for (int j = 0; j < 4; ++j) {
        y->qh[j] = (qh >> (8*j)) & 0xff;
    }
}

kernel void quantize_q5_1(device const float    * x  [[buffer(0)]],
                          device       blk_q5_1 * yy [[buffer(1)]],
                          constant   int64_t    & nb [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= (uint64_t) nb) return;
    device const float * xb = x + (uint64_t) i * QK;
    device blk_q5_1 * y = yy + i;

    float vmin = xb[0];
    float vmax = xb[0];
    for (int j = 1; j < QK; ++j) {
        const float v = xb[j];
        vmin = v < vmin ? v : vmin;
        vmax = v > vmax ? v : vmax;
    }

    const float d  = (vmax - vmin) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = (half) d;
    y->m = (half) vmin;

    uint32_t qh = 0;
    for (int j = 0; j < QK/2; ++j) {
        const float x0 = (xb[0    + j] - vmin)*id;
        const float x1 = (xb[QK/2 + j] - vmin)*id;
        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);
        y->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK/2);
    }
    for (int j = 0; j < 4; ++j) {
        y->qh[j] = (qh >> (8*j)) & 0xff;
    }
}

kernel void quantize_q8_0(device const float    * x  [[buffer(0)]],
                          device       blk_q8_0 * yy [[buffer(1)]],
                          constant   int64_t    & nb [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= (uint64_t) nb) return;
    device const float * xb = x + (uint64_t) i * QK;
    device blk_q8_0 * y = yy + i;

    float amax = 0.0f;
    for (int j = 0; j < QK; ++j) {
        amax = max(amax, fabs(xb[j]));
    }

    const float d  = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = (half) d;

    for (int j = 0; j < QK; ++j) {
        y->qs[j] = round(xb[j]*id);
    }
}

static inline int best_index_int8(int n, constant const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

kernel void quantize_iq4_nl(device const float      * x  [[buffer(0)]],
                            device       blk_iq4_nl * yy [[buffer(1)]],
                            constant   int64_t      & nb [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= (uint64_t) nb) return;
    device const float * xb = x + (uint64_t) i * QK;
    device blk_iq4_nl * y = yy + i;

    float amax = 0.0f;
    float vmax = 0.0f;
    for (int j = 0; j < QK; ++j) {
        const float v = xb[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            vmax = v;
        }
    }

    float d = vmax / k_iq4nl_values[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK/2; ++j) {
        const float x0 = xb[0    + j]*id;
        const float x1 = xb[QK/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, k_iq4nl_values, x0);
        const uint8_t xi1 = best_index_int8(16, k_iq4nl_values, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = k_iq4nl_values[xi0];
        const float v1 = k_iq4nl_values[xi1];
        const float w0 = xb[0    + j]*xb[0    + j];
        const float w1 = xb[QK/2 + j]*xb[QK/2 + j];
        sumqx += w0*v0*xb[j] + w1*v1*xb[QK/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = (half)(sumq2 > 0 ? sumqx/sumq2 : d);
}
)MSL";

constexpr int QK_SIMPLE = 32;

struct type_kernel {
    ggml_type    type;
    const char * kernel_name;
    size_t       block_bytes;   // output bytes per QK_SIMPLE input values (or per element for f16/bf16)
    bool         elementwise;
};

const type_kernel k_type_kernels[] = {
    { GGML_TYPE_F16,    "quantize_f16",    2,          true  },
    { GGML_TYPE_BF16,   "quantize_bf16",   2,          true  },
    { GGML_TYPE_Q4_0,   "quantize_q4_0",   2 + 16,     false },
    { GGML_TYPE_Q4_1,   "quantize_q4_1",   4 + 16,     false },
    { GGML_TYPE_Q5_0,   "quantize_q5_0",   2 + 4 + 16, false },
    { GGML_TYPE_Q5_1,   "quantize_q5_1",   4 + 4 + 16, false },
    { GGML_TYPE_Q8_0,   "quantize_q8_0",   2 + 32,     false },
    { GGML_TYPE_IQ4_NL, "quantize_iq4_nl", 2 + 16,     false },
};

const type_kernel * find_kernel(ggml_type type) {
    for (const type_kernel & k : k_type_kernels) {
        if (k.type == type) {
            return &k;
        }
    }
    return nullptr;
}

class metal_converter : public device_converter {
public:
    metal_converter() : name_("metal") {}

    bool init(std::string & error) {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) {
                error = "no Metal device available";
                return false;
            }
            NSError * ns_error   = nil;
            id<MTLLibrary> library = [device_ newLibraryWithSource:@(QZ_METAL_SOURCE) options:nil error:&ns_error];
            if (!library) {
                error = "failed to compile Metal kernels";
                if (ns_error) {
                    error += std::string(": ") + ns_error.localizedDescription.UTF8String;
                }
                return false;
            }
            for (size_t i = 0; i < sizeof(k_type_kernels) / sizeof(k_type_kernels[0]); ++i) {
                id<MTLFunction> fn = [library newFunctionWithName:@(k_type_kernels[i].kernel_name)];
                if (!fn) {
                    error = std::string("missing Metal kernel ") + k_type_kernels[i].kernel_name;
                    return false;
                }
                pipelines_[i] = [device_ newComputePipelineStateWithFunction:fn error:&ns_error];
                if (!pipelines_[i]) {
                    error = std::string("failed to create pipeline for ") + k_type_kernels[i].kernel_name;
                    if (ns_error) {
                        error += std::string(": ") + ns_error.localizedDescription.UTF8String;
                    }
                    return false;
                }
            }
            queue_ = [device_ newCommandQueue];
            if (!queue_) {
                error = "failed to create Metal command queue";
                return false;
            }
            return true;
        }
    }

    const std::string & name() const override {
        return name_;
    }

    bool supports(ggml_type dst_type) const override {
        return find_kernel(dst_type) != nullptr;
    }

    bool convert(const float * src, ggml_type dst_type, void * dst, int64_t nrows, int64_t n_per_row,
                 std::string & error) override {
        const type_kernel * kernel = find_kernel(dst_type);
        if (kernel == nullptr) {
            error = "unsupported type";
            return false;
        }
        size_t pipeline_index = kernel - k_type_kernels;

        // bound temporary buffer memory; a row is never split because
        // quantized blocks are row-oriented
        constexpr size_t max_chunk_input_bytes = 64u * 1024u * 1024u;
        const size_t  row_bytes = (size_t) n_per_row * sizeof(float);
        const size_t  dst_row_bytes =
            kernel->elementwise ? (size_t) n_per_row * kernel->block_bytes
                                : (size_t) n_per_row / QK_SIMPLE * kernel->block_bytes;
        const int64_t rows_per_chunk =
            std::max<int64_t>(1, std::min<int64_t>(nrows, max_chunk_input_bytes / std::max<size_t>(1, row_bytes)));

        @autoreleasepool {
            id<MTLBuffer> src_buf = [device_ newBufferWithLength:rows_per_chunk * row_bytes
                                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> dst_buf = [device_ newBufferWithLength:rows_per_chunk * dst_row_bytes
                                                         options:MTLResourceStorageModeShared];
            if (!src_buf || !dst_buf) {
                error = "failed to allocate Metal buffers";
                return false;
            }

            for (int64_t row0 = 0; row0 < nrows; row0 += rows_per_chunk) {
                const int64_t chunk_rows = std::min(rows_per_chunk, nrows - row0);
                const int64_t n          = chunk_rows * n_per_row;
                const int64_t n_items    = kernel->elementwise ? n : n / QK_SIMPLE;

                memcpy(src_buf.contents, src + row0 * n_per_row, n * sizeof(float));

                id<MTLCommandBuffer>         cmd = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                id<MTLComputePipelineState>  pso = pipelines_[pipeline_index];

                [enc setComputePipelineState:pso];
                [enc setBuffer:src_buf offset:0 atIndex:0];
                [enc setBuffer:dst_buf offset:0 atIndex:1];
                [enc setBytes:&n_items length:sizeof(n_items) atIndex:2];

                const NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
                [enc dispatchThreadgroups:MTLSizeMake((n_items + tg - 1) / tg, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];

                if (cmd.status != MTLCommandBufferStatusCompleted) {
                    error = "Metal command buffer failed";
                    if (cmd.error) {
                        error += std::string(": ") + cmd.error.localizedDescription.UTF8String;
                    }
                    return false;
                }

                memcpy((char *) dst + row0 * dst_row_bytes, dst_buf.contents, chunk_rows * dst_row_bytes);
            }
        }
        return true;
    }

private:
    std::string                 name_;
    id<MTLDevice>               device_ = nil;
    id<MTLCommandQueue>         queue_  = nil;
    id<MTLComputePipelineState> pipelines_[sizeof(k_type_kernels) / sizeof(k_type_kernels[0])] = {};
};

} // namespace

bool metal_device_available() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
}

std::string metal_device_description() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device ? device.name.UTF8String : "no Metal device";
    }
}

std::unique_ptr<device_converter> metal_device_open(std::string & error) {
    auto converter = std::make_unique<metal_converter>();
    if (!converter->init(error)) {
        return nullptr;
    }
    return converter;
}

} // namespace qz
