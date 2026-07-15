# quantizer

A lightweight, high-performance standalone tool for quantizing GGUF tensors. This utility allows you to take existing GGUF or safetensors models (e.g., F32, F16) and convert them into various quantized GGUF formats (Q4_K, Q8_0, etc.) or apply mixed-precision quantization using regex-based rules.

## Overview

The GGUF Quantizer is designed for precision and flexibility. Unlike standard one-size-fits-all quantization, this tool allows you to specify different quantization types for different tensors within the same model. This is particularly useful for preserving accuracy in sensitive layers (like attention mechanisms) while aggressively compressing less critical weights.

The quantizer is **fully self-contained**: it does not link against ggml (or any other library). The reference quantization kernels are vendored in `src/kernels/` (a verbatim copy of ggml's `ggml-quants.c` compiled against small local shim headers) and the GGUF container I/O is a from-scratch implementation in `src/gguf_io.cpp`. Safetensors input is supported via a from-scratch reader in `src/safetensors_io.cpp` (including its own minimal JSON parser - no external JSON library). Quantization runs on the CPU by default; optional accelerator backends (CUDA, ROCm/HIP, Metal) can be compiled in for faster conversion of the simple block formats.

## Architecture

```text
[ Input GGUF or safetensors ]  <-- format auto-detected by content
      |
[ GGUF Reader (src/gguf_io.cpp) ]              <-- parses header + tensor metadata,
[ safetensors Reader (src/safetensors_io.cpp) ]    passes GGUF KV metadata through verbatim
      |
      |-- [ Tensor Transformer ]
      |     |-- Rule Engine (regex matching for tensor names)
      |     |-- Type Resolver (determines target type per tensor)
      |     |-- Quantization Kernels (vendored ggml kernels, src/kernels/)
      |     |-- Device Layer (optional CUDA / ROCm / Metal kernels,
      |     |                 automatic CPU fallback per type)
      |     `-- Multi-threaded Executor (parallel processing of rows)
      |
[ Output GGUF ]  <-- same KV metadata (GGUF input) or no KVs (safetensors input),
                     new tensor types/offsets, aligned data
```

## Key Features

* **No ggml dependency**: everything needed is in this repository; the whole build is two small targets (a shared library and the CLI).
* **GGUF and safetensors input**: the input format is auto-detected by content; `.safetensors` files are converted to quantized GGUF in one step. Safetensors dtypes without a ggml equivalent are converted at load time (`F64` -> `f32`, `I64` -> `i32`, `F8_E4M3`/`F8_E5M2` -> `f16`, matching stable-diffusion.cpp); `U8` tensors are skipped.
* **Multi-part safetensors**: pass any part of a `model-00001-of-00003.safetensors` set and all parts are located by name and merged into a single output GGUF (duplicate tensor names across parts are rejected).
* **Mixed Precision Support**: use `--tensor-type-rules` to apply different quantization levels to specific tensors using regex patterns - with or without a global `--type`.
* **High Performance**: multi-threaded quantization engine; optional GPU acceleration.
* **Flexible Target Types**: supports a wide range of GGUF types, including K-quants (`Q4_K`, `Q5_K`), legacy formats, and newer experimental formats like `mxfp4`, `nvfp4`, etc.
* **Safety First**: automatically skips quantization for tensors that cannot be dequantized or where block alignment would be violated.
* **Metadata fidelity**: all key/value metadata is copied into the output byte-for-byte (safetensors has no KV metadata, so a GGUF produced from one contains tensor data only).

## Workflow

1.  **Initialization**: The tool detects the input format (GGUF magic vs. safetensors header) and parses its metadata (KV pairs and tensor list for GGUF; the JSON header for safetensors, with shapes mapped into ggml's dimension order).
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

Accelerators provide fast f32 -> `q4_0`/`q4_1`/`q5_0`/`q5_1`/`q8_0`/`iq4_nl`/`f16`/`bf16` kernels
(the same fast quantizers ggml uses on GPU); other target types automatically
fall back to the CPU kernels, with a one-time warning per type.

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
* **Standard Quants**: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q1_0`
* **K-Quants**: `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`
* **I-Quants**: `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq4_nl`, `iq4_xs`
* **T-Quants**: `tq1_0`, `tq2_0`
* **Experimental**: `mxfp4`, `nvfp4`

Notes:
* With a bare `--type`, 1D tensors (norms/biases) are kept at full precision, following llama.cpp convention; an explicit `--tensor-type-rules` match converts them anyway.
* GPU kernels use the same fast rounding as ggml's GPU copy kernels, so their output can differ slightly (but validly) from the CPU kernels for `q4_0`/`q4_1`/`q5_0`/`q5_1`/`iq4_nl`; CPU output is bit-for-bit identical to ggml's reference quantization.

## Updating the vendored kernels

The files `src/kernels/ggml-quants.c`, `src/kernels/ggml-quants.h` and
`src/kernels/ggml-common.h` are verbatim copies from ggml. To sync with a
newer ggml, simply re-copy those three files; the local shim headers
(`src/kernels/ggml.h`, `ggml-impl.h`, `ggml-cpu.h`, `ggml-cpu/ggml-cpu-impl.h`)
provide everything they need. If ggml adds a new type, extend the table in
`src/kernels/qz-traits.c` and the dispatch switches next to it.

## Acknowledgment

Thanks all contributors in the Community🤖, The authors of ggml (MIT License). Really appreciate!♥️

## License

This project is licensed under the MIT License. See `LICENSE`.
