// Thread-safe replacement for ggml.c's ggml_quantize_init/ggml_quantize_free.
// The iq2/iq3 grid setup in ggml-quants.c is not thread-safe by itself, so the
// calls are serialized here (ggml uses its own critical section for this).

#include "ggml.h"
#include "ggml-quants.h"

#include <mutex>

static std::mutex g_quantize_init_mutex;

extern "C" void ggml_quantize_init(enum ggml_type type) {
    std::lock_guard<std::mutex> lock(g_quantize_init_mutex);

    switch (type) {
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); break;
        case GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); break;
        case GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); break;
        default: // nothing
            break;
    }
}

extern "C" void ggml_quantize_free(void) {
    std::lock_guard<std::mutex> lock(g_quantize_init_mutex);

    iq2xs_free_impl(GGML_TYPE_IQ2_XXS);
    iq2xs_free_impl(GGML_TYPE_IQ2_XS);
    iq2xs_free_impl(GGML_TYPE_IQ2_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_M);
    iq3xs_free_impl(256);
    iq3xs_free_impl(512);
}
