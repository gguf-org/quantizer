# quantizer

A lightweight, high-performance standalone tool for quantizing GGUF tensors. This utility allows you to take existing GGUF or safetensors models (e.g., F32, F16) and convert them into various quantized GGUF formats (Q4_K, Q8_0, etc.) or apply mixed-precision quantization using regex-based rules.

## Overview

The GGUF Quantizer is designed for precision and flexibility. Unlike standard one-size-fits-all quantization, this tool allows you to specify different quantization types for different tensors within the same model. This is particularly useful for preserving accuracy in sensitive layers (like attention mechanisms) while aggressively compressing less critical weights.

The quantizer is **fully self-contained**: it links against no third-party library, and every line of it is this project's own code. The quantization kernels in `src/kernels/` are an independent implementation written against the GGUF on-disk formats (see [Quantization kernels](#quantization-kernels)), and the GGUF container I/O is a from-scratch implementation in `src/gguf_io.cpp`. Safetensors input is supported via a from-scratch reader in `src/safetensors_io.cpp` (including its own minimal JSON parser - no external JSON library). Quantization runs on the CPU by default; optional accelerator backends (CUDA, ROCm/HIP, Metal) can be compiled in for faster conversion of the simple block formats.

## Architecture

```text
[ Input GGUF or safetensors ]  <-- format auto-detected by content
      |
      v
[ GGUF Reader (src/gguf_io.cpp) ]              <-- parses header + tensor metadata,
[ safetensors Reader (src/safetensors_io.cpp) ]    passes GGUF KV metadata through verbatim
      |
      |-- [ Tensor Transformer ]
      |     |-- Rule Engine (regex matching for tensor names)
      |     |-- Type Resolver (determines target type per tensor)
      |     |-- Quantization Kernels (src/kernels/, this project's own)
      |     |-- Device Layer (optional CUDA / ROCm / Metal kernels,
      |     |                 automatic CPU fallback per type)
      |     `-- Multi-threaded Executor (parallel processing of rows)
      |
      v
[ Output GGUF ]  <-- same KV metadata (GGUF input) or no KVs (safetensors input),
                     new tensor types/offsets, aligned data
```

## Key Features

* **No third-party dependencies**: everything needed is in this repository; the whole build is two small targets (a shared library and the CLI).
* **GGUF and safetensors input**: the input format is auto-detected by content; `.safetensors` files are converted to quantized GGUF in one step. Safetensors dtypes with no GGUF equivalent are converted at load time (`F64` -> `f32`, `I64` -> `i32`, `F8_E4M3`/`F8_E5M2` -> `f16`, matching stable-diffusion.cpp); `U8` tensors are skipped.
* **Multi-part safetensors**: pass any part of a `model-00001-of-00003.safetensors` set and all parts are located by name and merged into a single output GGUF (duplicate tensor names across parts are rejected).
* **Mixed Precision Support**: use `--tensor-type-rules` to apply different quantization levels to specific tensors using regex patterns - with or without a global `--type`.
* **High Performance**: multi-threaded quantization engine; optional GPU acceleration.
* **Flexible Target Types**: supports a wide range of GGUF types, including K-quants (`Q4_K`, `Q5_K`), legacy formats, and newer experimental formats like `mxfp4`, `nvfp4`, etc.
* **Safety First**: automatically skips quantization for tensors that cannot be dequantized or where block alignment would be violated.
* **Metadata fidelity**: all key/value metadata is copied into the output byte-for-byte (safetensors has no KV metadata, so a GGUF produced from one contains tensor data only).

## Workflow

1.  **Initialization**: The tool detects the input format (GGUF magic vs. safetensors header) and parses its metadata (KV pairs and tensor list for GGUF; the JSON header for safetensors, with shapes mapped into GGUF's dimension order).
2.  **Planning**:
    *   The engine iterates through every tensor.
    *   It applies the default `--type` (if given) to all eligible tensors; without `--type`, tensors keep their original type unless a rule matches.
    *   It then checks for any overrides defined in `--tensor-type-rules` (first match wins).
    *   Validation is performed to ensure the target type is compatible with the source data (e.g., ensuring dequantizability and block alignment).
3.  **Execution**:
    *   The output GGUF header is written first.
    *   For each tensor, the tool seeks to the correct offset in the input file.
    *   Data is read into memory buffers and dequantized to F32 (multi-threaded).
    *   The quantization kernel processes the data - on the selected accelerator when it has a kernel for the target type, otherwise on the CPU across multiple threads.
    *   Quantized bytes are appended to the output file with proper alignment padding.

## Usage

### Requirements

* C++17 compatible compiler (GCC, Clang, or MSVC)
* CMake 3.15+
* optional, for accelerator builds: CUDA toolkit / ROCm (HIP) / macOS with Metal

### Clone the code
```bash
git clone https://github.com/gguf-org/quantizer
cd quantizer
```

### Building

CPU-only (default, no other dependencies):
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

With an accelerator backend:
```bash
cmake .. -DQUANTIZER_CUDA=ON    # NVIDIA GPUs (CUDA toolkit required)
cmake .. -DQUANTIZER_HIP=ON     # AMD GPUs (ROCm/HIP toolchain required)
cmake .. -DQUANTIZER_METAL=ON   # Apple GPUs (macOS only)
```

Accelerators provide f32 -> `q4_0`/`q4_1`/`q5_0`/`q5_1`/`q8_0`/`iq4_nl`/`f16`/`bf16`
kernels, which follow the same fits as the CPU kernels; other target types
automatically fall back to the CPU kernels, with a one-time warning per type.

### Command Line Interface

```bash
./quantizer -m <input_model.gguf|input_model.safetensors> -o <output_model.gguf> [--type <target_type>] [options]
```

#### Required Arguments:
* `-m, --model <file>`: Path to the source model, either GGUF or safetensors (auto-detected by content).
* `-o, --output <file>`: Path where the quantized GGUF will be saved.
* at least one of `--type` / `--tensor-type-rules`.

#### Options:
* `--type <type>`: The default quantization type for all eligible tensors (e.g., `q4_k`, `q8_0`, `f16`). Optional: when omitted, only tensors matched by `--tensor-type-rules` are converted and everything else is copied unchanged.
* `--tensor-type-rules "<regex>=<type>,..."`: Per-tensor type overrides based on tensor name regex (first match wins). Works standalone or together with `--type`, which it overrides for matching tensors.
    * *Example*: `--tensor-type-rules "layers.0.attention.weight=q8_0,layers.1.*=q4_k"`
* `--device <name>`: Accelerator to use: `cpu` (default), `auto` (best available), or a name shown by `--list-devices` (e.g. `cuda0`, `rocm0`, `metal`).
* `--list-devices`: List the devices available in this build and exit.
* `-t, --threads <n>`: Number of threads for quantization (defaults to hardware concurrency).
* `-h, --help`: Show the help message.

#### Example Commands

**Simple Quantization:**
Convert an F32 model to Q4_K.
```bash
./quantizer -m model-f32.gguf -o model-q4_k.gguf --type q4_k
```

**Safetensors Input:**
Convert a safetensors model directly to a quantized GGUF.
```bash
./quantizer -m model.safetensors -o model-q4_0.gguf --type q4_0
```

**Multi-part Safetensors Input:**
Pass any part; the sibling parts are found by name and merged into one GGUF.
```bash
./quantizer -m model-00001-of-00003.safetensors -o model-q4_0.gguf --type q4_0
```

**Rules-only Quantization (no default type):**
Quantize only the attention weights, leave everything else untouched.
```bash
./quantizer -m model.gguf -o model-out.gguf --tensor-type-rules "attention.*weight=q4_k"
```

**Mixed Precision Quantization:**
Set a global type of `q8_0`, but specifically use `q4_0` for all attention weights and `q5_0` for feed-forward layers.
```bash
./quantizer -m model.gguf -o model-mixed.gguf --type q8_0 \
  --tensor-type-rules "layers.*attention.*weight=q4_0,layers.*feed_forward.*weight=q5_0"
```

**GPU-accelerated Quantization:**
```bash
./quantizer -m model-f16.gguf -o model-q8_0.gguf --type q8_0 --device auto
```

## Supported Types

* **Floating Point**: `f32`, `f16`, `bf16`
* **Standard Quants**: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q1_0`, `q2_0`
* **K-Quants**: `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`
* **I-Quants**: `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq4_nl`, `iq4_xs`
* **T-Quants**: `tq1_0`, `tq2_0`
* **Experimental**: `mxfp4`, `nvfp4`

Notes:
* With a bare `--type`, 1D tensors (norms/biases) are kept at full precision, following llama.cpp convention; an explicit `--tensor-type-rules` match converts them anyway.
* GPU kernels run the same fits as the CPU ones, but device floating-point arithmetic can differ in the last bit, so their output may differ slightly (and validly) from the CPU kernels for `q4_0`/`q4_1`/`q5_0`/`q5_1`/`iq4_nl`.

## Quantization kernels

`src/kernels/` is an independent implementation of the GGUF tensor formats,
written against the on-disk layouts rather than derived from another
quantization library. It carries no third-party code and no third-party licence
obligations.

| file | contents |
| --- | --- |
| `qz_quant.h` | public interface: type ids, traits, encode/decode entry points |
| `qz_format.h` | on-disk block layouts, one struct per format, with the bit packing documented |
| `qz_fp.h` | f16, bf16, E8M0 and UE4M3 conversions |
| `qz_common.h` | the fitting primitives every encoder shares |
| `qz_pack.c` | block-scale formats: `q4_0`..`q8_0`, `q1_0`, ternary, MXFP4/NVFP4, `iq4_nl`/`iq4_xs` |
| `qz_super.c` | super-block formats: `q2_K`..`q6_K` |
| `qz_lattice.c` | codebook formats: `iq1_*`, `iq2_*`, `iq3_*` |
| `qz_decode.c` | decoders for every format |
| `qz_codebook.c` | the fixed codebook tables the lattice and FP4 formats decode against |
| `qz_traits.c` | type traits, dispatch, validation |

### How a format gets encoded

Most formats reduce to the same problem: pick a scale (and sometimes an offset)
so that a group of weights, rounded onto a small integer range, reconstructs as
closely as possible. Where a format fixes its own rule - "put the largest
magnitude at the end of the range, then round to nearest" - the encoder follows
that rule literally. Where the scale is open, it is fitted: start from a few
candidate scales, alternate between rounding at the current scale and
re-solving the scale for that rounding, and keep whichever result has the lowest
weighted error. The super-block formats add a second stage, re-solving the
shared f16 multipliers once the per-group integers are known, and the lattice
formats replace the rounding step with a nearest-neighbour search over their
codebook, accelerated by a table built at start-up.

Weights come from the importance matrix when one is supplied, multiplied by a
magnitude term; `qz_common.h` documents the trade-off and the compile-time
switches that move it.

### Codebook tables

`qz_codebook.c` holds the lattice grids (`iq1`, `iq2_*`, `iq3_*`) and the two
16-entry non-linear 4-bit tables. These are format constants, not implementation
choices: a block stores an index into them, so every conforming encoder and
decoder has to use exactly these values, the same way an image codec has to use
the quantization tables its format prescribes.

### Checking a change

`-DQUANTIZER_TOOLS=ON` builds two tools:

* `qz_selftest` - asserts, for every format, that the traits match the layouts,
  that encoded rows pass validation, that the round-trip error stays within
  what the format's width allows, and that requantizing settles instead of
  drifting. `ctest` runs it.
* `qz_bench` - encodes a deterministic matrix in every format and prints, as
  CSV, the encode time, the reconstruction error (plain and importance
  weighted) and a hash of the encoded bytes. Diff two runs to see what a change
  cost or bought.

Adding a format means: a struct in `qz_format.h`, an encoder, a decoder, a row
in the traits table in `qz_traits.c`, and its entries in the dispatch switches.

## License

This project is licensed under the MIT License. See `LICENSE`.

Everything in this repository is its own code, including the quantization
kernels: there is no vendored or ported third-party source, and the build links
no third-party library, so the MIT notice in `LICENSE` is the only one that
applies. What the kernels do share with every other implementation of these
formats is the formats themselves - block layouts, type ids, and the fixed
codebook tables in `qz_codebook.c` - which are what makes the output readable
by other GGUF tools and are not implementation choices to be made differently.
