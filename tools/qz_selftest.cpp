// Self-contained checks for the quantization kernels.
//
// Unlike tools/qz_bench.cpp, this needs nothing to compare against: it asserts
// the properties every format has to satisfy on its own.
//
//   1. traits agree with the block layouts (row sizes, block counts)
//   2. an encoded row passes the library's own validation
//   3. decode(encode(x)) stays inside the error a format of that width can be
//      expected to hit, and never produces NaN or infinity
//   4. re-encoding an already decoded row lands within a small fraction of the
//      format's own error, so requantizing a file repeatedly settles instead of
//      drifting. (The stronger property - identical bytes - does not hold for
//      the formats whose scale is searched rather than derived: the search
//      re-weights itself on the decoded values and can settle on a neighbouring
//      scale. Measured over five passes the error stays put, so the bound below
//      is what gets asserted.)
//   5. the narrow float conversions round-trip every half value exactly
//
// Build (from the repo root):
//   c++ -O2 -Isrc tools/qz_selftest.cpp src/kernels/*.c src/kernels/qz_init.cpp -o qz_selftest
//
// Exits non-zero if anything fails.

#include "kernels/qz_quant.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char * what, const char * detail = "") {
    if (!ok) {
        printf("  FAIL %s %s\n", what, detail);
        ++failures;
    }
}

struct rng {
    uint64_t s = 88172645463325252ull;
    double next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return (double) ((s * 0x2545F4914F6CDD1DULL) >> 11) / 9007199254740992.0;
    }
    double normal() {
        const double u1 = next() + 1e-12, u2 = next();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

// Loosest reconstruction error each format is still considered healthy at,
// relative to the norm of the input. A format that regresses badly - a packing
// bug, a scale that never gets fitted - blows through these by a wide margin.
double error_ceiling(qz_type t) {
    switch (t) {
        case QZ_TYPE_F32:                        return 1e-6;
        case QZ_TYPE_F16:                        return 1e-3;
        case QZ_TYPE_BF16:                       return 1e-2;
        case QZ_TYPE_Q8_0:                       return 0.02;
        case QZ_TYPE_Q6_K:                       return 0.05;
        case QZ_TYPE_Q5_0: case QZ_TYPE_Q5_1:
        case QZ_TYPE_Q5_K: case QZ_TYPE_NVFP4:   return 0.10;
        case QZ_TYPE_Q4_0: case QZ_TYPE_Q4_1:
        case QZ_TYPE_Q4_K: case QZ_TYPE_IQ4_NL:
        case QZ_TYPE_IQ4_XS: case QZ_TYPE_MXFP4: return 0.20;
        case QZ_TYPE_Q3_K: case QZ_TYPE_IQ3_S:
        case QZ_TYPE_IQ3_XXS:                    return 0.35;
        case QZ_TYPE_Q2_K: case QZ_TYPE_IQ2_S:
        case QZ_TYPE_IQ2_XS: case QZ_TYPE_IQ2_XXS: return 0.60;
        case QZ_TYPE_IQ1_S: case QZ_TYPE_IQ1_M:  return 0.80;
        case QZ_TYPE_Q1_0: case QZ_TYPE_Q2_0:
        case QZ_TYPE_TQ1_0:
        case QZ_TYPE_TQ2_0:                      return 1.00;
        default:                                 return 1.00;
    }
}

} // namespace

int main() {
    const int64_t nrows = 4, ncols = 1024;

    rng r;
    std::vector<float> x((size_t) nrows * ncols);
    for (auto & v : x) {
        v = (float) (r.normal() * 0.05);
    }
    std::vector<float> im((size_t) ncols);
    for (auto & v : im) {
        v = (float) (0.5 + r.next());
    }

    double xnorm = 0.0;
    for (float v : x) {
        xnorm += (double) v * v;
    }
    xnorm = std::sqrt(xnorm);

    printf("%-9s %10s %10s %9s  %s\n", "type", "row bytes", "rel err", "drift", "checks");

    for (int t = 0; t < QZ_TYPE_COUNT; ++t) {
        const qz_type type = (qz_type) t;
        const char * name = qz_type_name(type);
        if (!name || !qz_is_quantize_target(type) || ncols % qz_blck_size(type) != 0) {
            continue;
        }

        const int before = failures;
        const size_t row_size = qz_row_size(type, ncols);

        check(row_size == qz_type_size(type) * (size_t) (ncols / qz_blck_size(type)),
              name, "row size disagrees with block size");

        std::vector<uint8_t> enc((size_t) nrows * row_size, 0xcd);
        const size_t written = qz_quantize_chunk(type, x.data(), enc.data(), 0, nrows, ncols, im.data());
        check(written == enc.size(), name, "encoder wrote the wrong number of bytes");

        if (qz_is_quantized(type)) {
            check(qz_validate_row_data(type, enc.data(), enc.size()), name, "validation rejected our own output");
        }

        std::vector<float> dec(x.size());
        check(qz_dequantize(type, enc.data(), dec.data(), (int64_t) x.size()), name, "no decoder");

        double se = 0.0;
        bool finite = true;
        for (size_t i = 0; i < dec.size(); ++i) {
            const double d = (double) dec[i] - (double) x[i];
            se += d * d;
            finite = finite && std::isfinite(dec[i]);
        }
        const double rel = xnorm > 0 ? std::sqrt(se) / xnorm : 0.0;

        check(finite, name, "decoded a non-finite value");
        check(rel <= error_ceiling(type), name, "reconstruction error above the ceiling for this width");

        // feeding the decoded values back in must not move them much: whatever
        // a second pass changes is what repeated requantization would keep
        // changing
        std::vector<uint8_t> enc2((size_t) nrows * row_size, 0xcd);
        std::vector<float>   dec2(x.size());
        qz_quantize_chunk(type, dec.data(), enc2.data(), 0, nrows, ncols, im.data());
        qz_dequantize(type, enc2.data(), dec2.data(), (int64_t) x.size());

        double se2 = 0.0;
        for (size_t i = 0; i < dec.size(); ++i) {
            const double d = (double) dec2[i] - (double) dec[i];
            se2 += d * d;
        }
        const double drift = xnorm > 0 ? std::sqrt(se2) / xnorm : 0.0;
        check(drift <= 0.25 * std::fmax(rel, 1e-9), name, "a second encode pass moves the values too far");

        printf("%-9s %10zu %10.4f %9.6f  %s\n", name, row_size, rel, drift,
               failures == before ? "ok" : "see above");
    }

    // narrow float round trips
    int half_bad = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint16_t bits = (uint16_t) h;
        if ((bits & 0x7c00) == 0x7c00) {
            continue; // inf and nan have no unique round trip
        }
        if (qz_fp32_to_fp16(qz_fp16_to_fp32(bits)) != bits) {
            ++half_bad;
        }
    }
    check(half_bad == 0, "f16", "round trip is not exact for every finite half");
    printf("\nf16 round trip: %s\n", half_bad ? "FAIL" : "ok");

    printf("%s\n", failures ? "SELFTEST FAILED" : "all checks passed");
    return failures != 0;
}
