# quantizer

A lightweight, high-performance standalone tool for quantizing GGUF tensors. This utility allows you to take existing GGUF models (e.g., F32, F16) and convert them into various quantized formats (Q4_K, Q8_0, etc.) or apply mixed-precision quantization using regex-based rules.

## Overview

The GGUF Quantizer is designed for precision and flexibility. Unlike standard one-size-fits-all quantization, this tool allows you to specify different quantization types for different tensors within the same model. This is particularly useful for preserving accuracy in sensitive layers (like attention mechanisms) while aggressively compressing less critical weights.

## Architecture

The project is built on top of the `ggml` and `gguf-connector` libraries, providing a streamlined pipeline for tensor manipulation.

```text
[ Input GGUF ] 
      |
      v
[ Metadata Parser ]  <-- Reads GGUF header, KV pairs, and tensor metadata
      |
      |-- [ Tensor Transformer ]
      |     |-- Rule Engine (Regex matching for tensor names)
      |     |-- Type Resolver (Determines target type per tensor)
      |     |-- Quantization Kernel (wraps ggml_quantize_chunk)
      |     `-- Multi-threaded Executor (Parallel processing of rows)
      |
      v
[ Output GGUF ]      <-- Reconstructs GGUF with new weights and alignment
```

## Key Features

* **Mixed Precision Support**: Use `--tensor-type-rules` to apply different quantization levels to specific tensors using regex patterns.
* **High Performance**: Multi-threaded quantization engine utilizing `ggml` kernels.
* **Flexible Target Types**: Supports a wide range of GGUF types, including K-quants (`Q4_K`, `Q5_K`), legacy formats, and newer experimental formats like `mxfp4`, `nvfp4`, etc.
* **Safety First**: Automatically skip quantization for tensors that cannot be dequantized or where block alignment would be violated.
* **Minimal Overhead**: A standalone CLI tool with no heavy dependencies outside of the GGML/GGUF ecosystem.

## Workflow

1.  **Initialization**: The tool loads the input GGUF file and parses its metadata (KV pairs and tensor list).
2.  **Planning**: 
    *   The engine iterates through every tensor.
    *   It applies the default `--type` to all tensors.
    *   It then checks for any overrides defined in `--tensor-type-rules`.
    *   Validation is performed to ensure the target type is compatible with the source data (e.g., ensuring dequantizability).
3.  **Execution**:
    *   The output GGUF header is written first.
    *   For each tensor, the tool seeks to the correct offset in the input file.
    *   Data is read into memory buffers.
    *   The quantization kernel processes the data (optionally across multiple threads).
    *   Quantized bytes are appended to the output file with proper alignment padding.

## Usage

### Requirements

* C++17 compatible compiler (GCC, Clang, or MSVC)
* CMake 3.10+

### Clone the code
```bash
git clone --recursive https://github.com/gguf-org/quantizer
cd quantizer
```

### Building

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### Command Line Interface

```bash
./quantizer -m <input_model.gguf> -o <output_model.gguf> --type <target_type> [options]
```

#### Required Arguments:
* `-m, --model <file>`: Path to the source GGUF file.
* `-o, --output <file>`: Path where the quantized GGUF will be saved.
* `--type <type>`: The default quantization type for all tensors (e.g., `q4_k`, `q8_0`, `f16`).

#### Options:
* `--tensor-type-rules "<regex>=<type>,..."`: Overrides the default type based on tensor name regex.
    * *Example*: `--tensor-type-rules "layers.0.attention.weight=q8_0,layers.1.*=q4_k"`
* `-t, --threads <n>`: Number of threads for quantization (defaults to hardware concurrency).
* `-h, --help`: Show the help message.

#### Example Commands

**Simple Quantization:**
Convert an F32 model to Q4_K.
```bash
./quantizer -m model-f32.gguf -o model-q4_k.gguf --type q4_k
```

**Mixed Precision Quantization:**
Set a global type of `q8_0`, but specifically use `q4_0` for all attention weights and `q5_0` for feed-forward layers.
```bash
./quantizer -m model.gguf -o model-mixed.gguf --type q8_0 \
  --tensor-type-rules "layers.*attention.*weight=q4_0,layers.*feed_forward.*weight=q5_0"
```

## Supported Types

The tool supports virtually all types supported by the current `ggml` implementation, including:
* **Floating Point**: `f32`, `f16`, `bf16`
* **Standard Quants**: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q1_0`
* **K-Quants**: `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`
* **I-Quants**: `iq1_s`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq4_nl`, `iq4_xs`
* **T-Quants**: `tq1_0`, `tq2_0`
* **Experimental**: `mxfp4`, `nvfp4`

## License

This project is licensed under the MIT License. See `LICENSE`.