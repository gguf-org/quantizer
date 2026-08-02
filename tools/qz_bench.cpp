// Round-trip quality harness for the quantization kernels.
//
// For every type the quantizer can emit, this encodes a deterministic test
// matrix, decodes it again and prints the reconstruction error together with a
// hash of the encoded bytes. Running it against two builds gives a direct
// comparison: the error columns show whether a kernel change costs accuracy,
// the hash column shows whether the encoded bytes moved at all.
//
// Build (from the repo root):
//   c++ -O2 -Isrc tools/qz_bench.cpp src/kernels/*.c -o qz_bench
//
// Output is CSV on stdout so runs can be diffed directly.

#include "kernels/qz_quant.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// xorshift64* - deterministic across platforms and compilers, which matters
// because two builds must see bit-identical input to be comparable.
struct rng {
    uint64_t s;

    explicit rng(uint64_t seed) : s(seed ? seed : 1) {}

    uint64_t next_u64() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    }

    // uniform in [0,1)
    double next_unit() {
        return (double) (next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }

    // standard normal (Box-Muller, one value per call is fine here)
    double next_normal() {
        const double u1 = next_unit() + 1e-12;
        const double u2 = next_unit();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

// A weight-like test matrix: mostly normal, with a per-row scale spread and a
// sprinkle of outliers, which is what separates good from bad scale search.
std::vector<float> make_data(int64_t nrows, int64_t n_per_row, uint64_t seed) {
    rng r(seed);
    std::vector<float> x((size_t) nrows * n_per_row);

    for (int64_t i = 0; i < nrows; ++i) {
        const double row_scale = 0.02 * std::exp(2.0 * r.next_unit());
        for (int64_t j = 0; j < n_per_row; ++j) {
            double v = r.next_normal() * row_scale;
            if (r.next_unit() < 0.01) {
                v *= 8.0; // outlier
            }
            x[(size_t) i * n_per_row + j] = (float) v;
        }
    }
    return x;
}

// Column importances, shaped like a real imatrix (positive, wide dynamic range).
std::vector<float> make_imatrix(int64_t n_per_row, uint64_t seed) {
    rng r(seed);
    std::vector<float> w((size_t) n_per_row);
    for (int64_t j = 0; j < n_per_row; ++j) {
        w[(size_t) j] = (float) (0.1 + std::exp(3.0 * r.next_unit()));
    }
    return w;
}

uint64_t fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

struct result {
    double rmse;
    double max_err;
    double rel_rmse; // rmse / rms(x), the scale-free number to compare across types
    double wrel;     // same, but weighted by column importance: what an imatrix run optimizes
    double ms;       // encode time for this run
    uint64_t hash;
    size_t bytes;
};

bool run_type(qz_type type, const std::vector<float> & x, const std::vector<float> & imatrix,
              int64_t nrows, int64_t n_per_row, result & out) {
    if (!qz_is_quantize_target(type)) {
        return false;
    }
    if (n_per_row % qz_blck_size(type) != 0) {
        return false;
    }

    const size_t row_size = qz_row_size(type, n_per_row);
    std::vector<uint8_t> enc((size_t) nrows * row_size);

    if (imatrix.empty() && qz_quantize_requires_imatrix(type)) {
        return false; // nothing to compare: this type cannot be encoded blind
    }
    const float * im = imatrix.empty() ? nullptr : imatrix.data();

    qz_quantize_init(type); // keep table building out of the timing

    const auto t0 = std::chrono::steady_clock::now();
    const size_t written = qz_quantize_chunk(type, x.data(), enc.data(), 0, nrows, n_per_row, im);
    const auto t1 = std::chrono::steady_clock::now();
    out.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (written != enc.size()) {
        return false;
    }

    std::vector<float> dec(x.size());
    if (!qz_dequantize(type, enc.data(), dec.data(), (int64_t) x.size())) {
        return false;
    }

    double se = 0.0, sx = 0.0, mx = 0.0, wse = 0.0, wsx = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double d = (double) dec[i] - (double) x[i];
        const double w = imatrix.empty() ? 1.0 : (double) imatrix[i % (size_t) n_per_row];
        se += d * d;
        sx += (double) x[i] * (double) x[i];
        wse += w * d * d;
        wsx += w * (double) x[i] * (double) x[i];
        mx = std::fmax(mx, std::fabs(d));
    }

    out.rmse     = std::sqrt(se / (double) x.size());
    out.max_err  = mx;
    out.rel_rmse = sx > 0.0 ? std::sqrt(se / sx) : 0.0;
    out.wrel     = wsx > 0.0 ? std::sqrt(wse / wsx) : 0.0;
    out.hash     = fnv1a(enc.data(), enc.size());
    out.bytes    = enc.size();
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    int64_t nrows     = 32;
    int64_t n_per_row = 1024;
    uint64_t seed     = 20260802;
    bool use_imatrix  = true;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) {
            nrows = atoll(argv[++i]);
        } else if (a == "--cols" && i + 1 < argc) {
            n_per_row = atoll(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            seed = strtoull(argv[++i], nullptr, 10);
        } else if (a == "--no-imatrix") {
            use_imatrix = false;
        } else {
            fprintf(stderr, "usage: %s [--rows N] [--cols N] [--seed N] [--no-imatrix]\n", argv[0]);
            return 1;
        }
    }

    const std::vector<float> x  = make_data(nrows, n_per_row, seed);
    const std::vector<float> im = use_imatrix ? make_imatrix(n_per_row, seed ^ 0x9E3779B9ULL)
                                              : std::vector<float>();

    printf("type,bytes,ms,rmse,max_err,rel_rmse,wrel,hash\n");

    for (int t = 0; t < QZ_TYPE_COUNT; ++t) {
        const qz_type type = (qz_type) t;
        const char * name  = qz_type_name(type);
        if (!name) {
            continue;
        }
        result r;
        if (!run_type(type, x, im, nrows, n_per_row, r)) {
            continue;
        }
        printf("%s,%zu,%.2f,%.6e,%.6e,%.6e,%.6e,%016llx\n", name, r.bytes, r.ms, r.rmse, r.max_err, r.rel_rmse,
               r.wrel, (unsigned long long) r.hash);
    }

    return 0;
}
