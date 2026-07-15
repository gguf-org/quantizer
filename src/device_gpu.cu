// CUDA / ROCm(HIP) accelerator backend: fast f32 -> quant kernels for the
// simple block formats (ported from ggml's CUDA cpy kernels) plus f16/bf16.
// K-quants and iq-quants have no GPU kernels and fall back to the CPU path.
//
// The same file compiles as CUDA (default) or HIP (with QZ_USE_HIP defined and
// the source language set to HIP in CMake).

#include "device_gpu.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(QZ_USE_HIP)
#  include <hip/hip_runtime.h>
#  include <hip/hip_fp16.h>
#  define gpuDeviceProp           hipDeviceProp_t
#  define gpuError_t              hipError_t
#  define gpuFree                 hipFree
#  define gpuGetDeviceCount       hipGetDeviceCount
#  define gpuGetDeviceProperties  hipGetDeviceProperties
#  define gpuGetErrorString       hipGetErrorString
#  define gpuGetLastError         hipGetLastError
#  define gpuMalloc               hipMalloc
#  define gpuMemcpyAsync          hipMemcpyAsync
#  define gpuMemcpyDeviceToHost   hipMemcpyDeviceToHost
#  define gpuMemcpyHostToDevice   hipMemcpyHostToDevice
#  define gpuSetDevice            hipSetDevice
#  define gpuStreamCreate         hipStreamCreate
#  define gpuStreamDestroy        hipStreamDestroy
#  define gpuStreamSynchronize    hipStreamSynchronize
#  define gpuStream_t             hipStream_t
#  define gpuSuccess              hipSuccess
#  define QZ_GPU_NAME_PREFIX      "rocm"
#else
#  include <cuda_runtime.h>
#  include <cuda_fp16.h>
#  define gpuDeviceProp           cudaDeviceProp
#  define gpuError_t              cudaError_t
#  define gpuFree                 cudaFree
#  define gpuGetDeviceCount       cudaGetDeviceCount
#  define gpuGetDeviceProperties  cudaGetDeviceProperties
#  define gpuGetErrorString       cudaGetErrorString
#  define gpuGetLastError         cudaGetLastError
#  define gpuMalloc               cudaMalloc
#  define gpuMemcpyAsync          cudaMemcpyAsync
#  define gpuMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#  define gpuMemcpyHostToDevice   cudaMemcpyHostToDevice
#  define gpuSetDevice            cudaSetDevice
#  define gpuStreamCreate         cudaStreamCreate
#  define gpuStreamDestroy        cudaStreamDestroy
#  define gpuStreamSynchronize    cudaStreamSynchronize
#  define gpuStream_t             cudaStream_t
#  define gpuSuccess              cudaSuccess
#  define QZ_GPU_NAME_PREFIX      "cuda"
#endif

namespace qz {
namespace {

// ------------------------------------------------------------------
// block layouts (must match ggml-common.h bit-for-bit)
// ------------------------------------------------------------------

constexpr int QK4_0  = 32;
constexpr int QK4_1  = 32;
constexpr int QK5_0  = 32;
constexpr int QK5_1  = 32;
constexpr int QK8_0  = 32;
constexpr int QK4_NL = 32;

struct blk_q4_0 { uint16_t d;              uint8_t qs[QK4_0 / 2]; };
struct blk_q4_1 { uint16_t d; uint16_t m;  uint8_t qs[QK4_1 / 2]; };
struct blk_q5_0 { uint16_t d;              uint8_t qh[4]; uint8_t qs[QK5_0 / 2]; };
struct blk_q5_1 { uint16_t d; uint16_t m;  uint8_t qh[4]; uint8_t qs[QK5_1 / 2]; };
struct blk_q8_0 { uint16_t d;              int8_t  qs[QK8_0]; };
struct blk_iq4_nl { uint16_t d;            uint8_t qs[QK4_NL / 2]; };

static_assert(sizeof(blk_q4_0)   == 2 + QK4_0 / 2,     "wrong q4_0 block size");
static_assert(sizeof(blk_q4_1)   == 4 + QK4_1 / 2,     "wrong q4_1 block size");
static_assert(sizeof(blk_q5_0)   == 2 + 4 + QK5_0 / 2, "wrong q5_0 block size");
static_assert(sizeof(blk_q5_1)   == 4 + 4 + QK5_1 / 2, "wrong q5_1 block size");
static_assert(sizeof(blk_q8_0)   == 2 + QK8_0,         "wrong q8_0 block size");
static_assert(sizeof(blk_iq4_nl) == 2 + QK4_NL / 2,    "wrong iq4_nl block size");

__constant__ int8_t k_iq4nl_values[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                                              1,   13,  25,  38,  53,  69,  89, 113 };

__device__ __forceinline__ uint16_t f32_to_f16_bits(float f) {
    return __half_as_ushort(__float2half(f));
}

// round-to-nearest-even fp32 -> bf16, identical to the CPU reference
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffff) > 0x7f800000) { // nan
        return (uint16_t) ((u >> 16) | 64); // force to quiet
    }
    return (uint16_t) ((u + (0x7fff + ((u >> 16) & 1))) >> 16);
}

// ------------------------------------------------------------------
// per-block quantizers (ported from ggml-cuda/cpy-utils.cuh)
// ------------------------------------------------------------------

__device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

__device__ void quantize_block_q4_0(const float * x, blk_q4_0 * y) {
    float amax = 0.0f;
    float vmax = 0.0f;
    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = f32_to_f16_bits(d);

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;
        const uint8_t xi0 = min(15, (int)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int)(x1 + 8.5f));
        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

__device__ void quantize_block_q4_1(const float * x, blk_q4_1 * y) {
    float vmin =  FLT_MAX;
    float vmax = -FLT_MAX;
    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = f32_to_f16_bits(d);
    y->m = f32_to_f16_bits(vmin);

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;
        const uint8_t xi0 = min(15, (int)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int)(x1 + 0.5f));
        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

__device__ void quantize_block_q5_0(const float * x, blk_q5_0 * y) {
    float amax = 0.0f;
    float vmax = 0.0f;
    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = f32_to_f16_bits(d);

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;
        const uint8_t xi0 = min(31, (int)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int)(x1 + 16.5f));
        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

__device__ void quantize_block_q5_1(const float * x, blk_q5_1 * y) {
    float vmin = x[0];
    float vmax = x[0];
    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        vmin = v < vmin ? v : vmin;
        vmax = v > vmax ? v : vmax;
    }

    const float d  = (vmax - vmin) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = f32_to_f16_bits(d);
    y->m = f32_to_f16_bits(vmin);

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK5_1/2 + j] - vmin)*id;
        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);
        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

__device__ void quantize_block_q8_0(const float * x, blk_q8_0 * y) {
    float amax = 0.0f; // absolute max
    for (int j = 0; j < QK8_0; j++) {
        amax = fmaxf(amax, fabsf(x[j]));
    }

    const float d  = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = f32_to_f16_bits(d);

    for (int j = 0; j < QK8_0; ++j) {
        y->qs[j] = roundf(x[j]*id);
    }
}

__device__ void quantize_block_iq4_nl(const float * x, blk_iq4_nl * y) {
    float amax = 0.0f;
    float vmax = 0.0f;
    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / k_iq4nl_values[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, k_iq4nl_values, x0);
        const uint8_t xi1 = best_index_int8(16, k_iq4nl_values, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = k_iq4nl_values[xi0];
        const float v1 = k_iq4nl_values[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = f32_to_f16_bits(sumq2 > 0 ? sumqx/sumq2 : d);
}

// ------------------------------------------------------------------
// kernels: one thread per quant block / element
// ------------------------------------------------------------------

template <typename block_t, int qk, void (*quantize_block)(const float *, block_t *)>
__global__ void k_quantize_blocks(const float * __restrict__ x, block_t * __restrict__ y, int64_t nblocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nblocks) {
        quantize_block(x + i * qk, y + i);
    }
}

__global__ void k_f32_to_f16(const float * __restrict__ x, uint16_t * __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = f32_to_f16_bits(x[i]);
    }
}

__global__ void k_f32_to_bf16(const float * __restrict__ x, uint16_t * __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = f32_to_bf16_bits(x[i]);
    }
}

size_t gpu_row_size(ggml_type type, int64_t n_per_row) {
    switch (type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:   return (size_t) n_per_row * 2;
        case GGML_TYPE_Q4_0:   return (size_t) n_per_row / QK4_0  * sizeof(blk_q4_0);
        case GGML_TYPE_Q4_1:   return (size_t) n_per_row / QK4_1  * sizeof(blk_q4_1);
        case GGML_TYPE_Q5_0:   return (size_t) n_per_row / QK5_0  * sizeof(blk_q5_0);
        case GGML_TYPE_Q5_1:   return (size_t) n_per_row / QK5_1  * sizeof(blk_q5_1);
        case GGML_TYPE_Q8_0:   return (size_t) n_per_row / QK8_0  * sizeof(blk_q8_0);
        case GGML_TYPE_IQ4_NL: return (size_t) n_per_row / QK4_NL * sizeof(blk_iq4_nl);
        default:               return 0;
    }
}

class gpu_converter : public device_converter {
public:
    gpu_converter(int device, std::string name) : device_(device), name_(std::move(name)) {}

    ~gpu_converter() override {
        if (d_src_) gpuFree(d_src_);
        if (d_dst_) gpuFree(d_dst_);
        if (stream_) gpuStreamDestroy(stream_);
    }

    bool init(std::string & error) {
        if (!check(gpuSetDevice(device_), "set device", error)) {
            return false;
        }
        return check(gpuStreamCreate(&stream_), "create stream", error);
    }

    const std::string & name() const override {
        return name_;
    }

    bool supports(ggml_type dst_type) const override {
        return gpu_row_size(dst_type, 32) != 0;
    }

    bool convert(const float * src, ggml_type dst_type, void * dst, int64_t nrows, int64_t n_per_row,
                 std::string & error) override {
        if (!supports(dst_type)) {
            error = "unsupported type";
            return false;
        }
        if (!check(gpuSetDevice(device_), "set device", error)) {
            return false;
        }

        // bound temporary device memory; a row is never split because
        // quantized blocks are row-oriented
        constexpr size_t max_chunk_input_bytes = 64u * 1024u * 1024u;
        const size_t  row_bytes     = (size_t) n_per_row * sizeof(float);
        const size_t  dst_row_bytes = gpu_row_size(dst_type, n_per_row);
        const int64_t rows_per_chunk =
            std::max<int64_t>(1, std::min<int64_t>(nrows, max_chunk_input_bytes / std::max<size_t>(1, row_bytes)));

        if (!reserve(rows_per_chunk * row_bytes, rows_per_chunk * dst_row_bytes, error)) {
            return false;
        }

        for (int64_t row0 = 0; row0 < nrows; row0 += rows_per_chunk) {
            const int64_t chunk_rows = std::min(rows_per_chunk, nrows - row0);
            const int64_t n          = chunk_rows * n_per_row;

            if (!check(gpuMemcpyAsync(d_src_, src + row0 * n_per_row, n * sizeof(float), gpuMemcpyHostToDevice,
                                      stream_), "copy to device", error)) {
                return false;
            }

            constexpr int threads = 256;
            switch (dst_type) {
                case GGML_TYPE_F16:
                    k_f32_to_f16<<<(n + threads - 1) / threads, threads, 0, stream_>>>(
                        (const float *) d_src_, (uint16_t *) d_dst_, n);
                    break;
                case GGML_TYPE_BF16:
                    k_f32_to_bf16<<<(n + threads - 1) / threads, threads, 0, stream_>>>(
                        (const float *) d_src_, (uint16_t *) d_dst_, n);
                    break;
                case GGML_TYPE_Q4_0:
                    launch_blocks<blk_q4_0, QK4_0, quantize_block_q4_0>(n);
                    break;
                case GGML_TYPE_Q4_1:
                    launch_blocks<blk_q4_1, QK4_1, quantize_block_q4_1>(n);
                    break;
                case GGML_TYPE_Q5_0:
                    launch_blocks<blk_q5_0, QK5_0, quantize_block_q5_0>(n);
                    break;
                case GGML_TYPE_Q5_1:
                    launch_blocks<blk_q5_1, QK5_1, quantize_block_q5_1>(n);
                    break;
                case GGML_TYPE_Q8_0:
                    launch_blocks<blk_q8_0, QK8_0, quantize_block_q8_0>(n);
                    break;
                case GGML_TYPE_IQ4_NL:
                    launch_blocks<blk_iq4_nl, QK4_NL, quantize_block_iq4_nl>(n);
                    break;
                default:
                    error = "unsupported type";
                    return false;
            }
            if (!check(gpuGetLastError(), "launch kernel", error)) {
                return false;
            }

            if (!check(gpuMemcpyAsync((char *) dst + row0 * dst_row_bytes, d_dst_, chunk_rows * dst_row_bytes,
                                      gpuMemcpyDeviceToHost, stream_), "copy from device", error)) {
                return false;
            }
            if (!check(gpuStreamSynchronize(stream_), "synchronize", error)) {
                return false;
            }
        }
        return true;
    }

private:
    template <typename block_t, int qk, void (*fn)(const float *, block_t *)>
    void launch_blocks(int64_t n) {
        constexpr int threads = 256;
        const int64_t nblocks = n / qk;
        k_quantize_blocks<block_t, qk, fn><<<(nblocks + threads - 1) / threads, threads, 0, stream_>>>(
            (const float *) d_src_, (block_t *) d_dst_, nblocks);
    }

    bool reserve(size_t src_bytes, size_t dst_bytes, std::string & error) {
        if (src_bytes > src_cap_) {
            if (d_src_) gpuFree(d_src_);
            d_src_   = nullptr;
            src_cap_ = 0;
            if (!check(gpuMalloc(&d_src_, src_bytes), "allocate device memory", error)) {
                return false;
            }
            src_cap_ = src_bytes;
        }
        if (dst_bytes > dst_cap_) {
            if (d_dst_) gpuFree(d_dst_);
            d_dst_   = nullptr;
            dst_cap_ = 0;
            if (!check(gpuMalloc(&d_dst_, dst_bytes), "allocate device memory", error)) {
                return false;
            }
            dst_cap_ = dst_bytes;
        }
        return true;
    }

    static bool check(gpuError_t err, const char * what, std::string & error) {
        if (err == gpuSuccess) {
            return true;
        }
        error = std::string(what) + ": " + gpuGetErrorString(err);
        return false;
    }

    int         device_;
    std::string name_;
    gpuStream_t stream_ = nullptr;
    void *      d_src_  = nullptr;
    void *      d_dst_  = nullptr;
    size_t      src_cap_ = 0;
    size_t      dst_cap_ = 0;
};

} // namespace

int gpu_device_count() {
    int count = 0;
    if (gpuGetDeviceCount(&count) != gpuSuccess) {
        return 0;
    }
    return count;
}

std::string gpu_device_name(int index) {
    return QZ_GPU_NAME_PREFIX + std::to_string(index);
}

std::string gpu_device_description(int index) {
    gpuDeviceProp prop;
    if (gpuGetDeviceProperties(&prop, index) != gpuSuccess) {
        return "unknown GPU";
    }
    return prop.name;
}

std::unique_ptr<device_converter> gpu_device_open(int index, std::string & error) {
    auto converter = std::make_unique<gpu_converter>(index, gpu_device_name(index));
    if (!converter->init(error)) {
        return nullptr;
    }
    return converter;
}

} // namespace qz
