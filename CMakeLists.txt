# 3.15+ is required so that CMP0091 defaults to NEW, which makes the
# MSVC_RUNTIME_LIBRARY / CMAKE_MSVC_RUNTIME_LIBRARY settings below actually
# take effect (on 3.14 they are silently ignored and MSVC falls back to the
# dynamic CRT baked into CMAKE_<LANG>_FLAGS_<CONFIG>).
cmake_minimum_required(VERSION 3.15)
project(gguf-quantizer C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# --------------------------------------------------------------------------
# Options
#
# The quantizer is fully self-contained: the quantization kernels are vendored
# in src/kernels/ (extracted from ggml) and the GGUF I/O is implemented in
# src/gguf_io.cpp - there is no ggml dependency. The CPU path is always built;
# accelerators are opt-in:
#   -DQUANTIZER_CUDA=ON    NVIDIA GPUs (requires the CUDA toolkit)
#   -DQUANTIZER_HIP=ON     AMD GPUs / ROCm (requires the HIP toolchain)
#   -DQUANTIZER_METAL=ON   Apple GPUs (macOS only)
# --------------------------------------------------------------------------
option(QUANTIZER_CUDA  "build with CUDA support (NVIDIA GPUs)" OFF)
option(QUANTIZER_HIP   "build with HIP/ROCm support (AMD GPUs)" OFF)
option(QUANTIZER_METAL "build with Metal support (Apple GPUs)" OFF)

if(QUANTIZER_CUDA AND QUANTIZER_HIP)
    message(FATAL_ERROR "QUANTIZER_CUDA and QUANTIZER_HIP are mutually exclusive")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()

# --------------------------------------------------------------------------
# Make the build self-contained so the contents of bin/ can be copied to a
# machine that never had a compiler/redistributable installed:
#  - MSVC: statically link the CRT (no MSVCP140.dll/VCRUNTIME140*.dll/
#    api-ms-win-crt-*.dll dependency).
#  - ELF/Mach-O: point RPATH at the executable's own directory ($ORIGIN /
#    @loader_path) instead of the absolute build-tree path, and bake that
#    into the binaries even without running `cmake --install`.
# --------------------------------------------------------------------------
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

# --------------------------------------------------------------------------
# gguf_quantizer: shared library with the reusable quantization logic
# --------------------------------------------------------------------------
set(QUANTIZER_SOURCES
    src/quantize.cpp
    src/quantize.h
    src/gguf_io.cpp
    src/gguf_io.h
    src/safetensors_io.cpp
    src/safetensors_io.h
    src/device.cpp
    src/device.h
    src/kernels/ggml-quants.c   # vendored from ggml (verbatim)
    src/kernels/qz-traits.c
    src/kernels/qz-init.cpp
)

if(QUANTIZER_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # cover common consumer GPUs from Maxwell (52) through Hopper/Ada (90)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90;100;120")
    endif()
    list(APPEND QUANTIZER_SOURCES src/device_gpu.cu src/device_gpu.h)
endif()

if(QUANTIZER_HIP)
    enable_language(HIP)
    list(APPEND QUANTIZER_SOURCES src/device_gpu.cu src/device_gpu.h)
    set_source_files_properties(src/device_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(QUANTIZER_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "QUANTIZER_METAL requires macOS")
    endif()
    enable_language(OBJCXX)
    list(APPEND QUANTIZER_SOURCES src/device_metal.mm src/device_metal.h)
    set_source_files_properties(src/device_metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()

add_library(gguf_quantizer SHARED ${QUANTIZER_SOURCES})
target_include_directories(gguf_quantizer PUBLIC src)
target_compile_definitions(gguf_quantizer PRIVATE GGUF_QUANTIZER_BUILD PUBLIC GGUF_QUANTIZER_SHARED)

if(QUANTIZER_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(gguf_quantizer PRIVATE QZ_USE_CUDA)
    target_link_libraries(gguf_quantizer PRIVATE CUDA::cudart)
endif()

if(QUANTIZER_HIP)
    find_package(hip REQUIRED)
    target_compile_definitions(gguf_quantizer PRIVATE QZ_USE_HIP)
    target_link_libraries(gguf_quantizer PRIVATE hip::host)
endif()

if(QUANTIZER_METAL)
    target_compile_definitions(gguf_quantizer PRIVATE QZ_USE_METAL)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    target_link_libraries(gguf_quantizer PRIVATE ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK})
endif()

find_package(Threads REQUIRED)
target_link_libraries(gguf_quantizer PRIVATE Threads::Threads)

set_target_properties(gguf_quantizer PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# portable optimization baseline for the scalar kernels; the hot loops
# vectorize well under -O3 without requiring -march=native
if(NOT MSVC)
    target_compile_options(gguf_quantizer PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()

# --------------------------------------------------------------------------
# quantizer: CLI front-end
# --------------------------------------------------------------------------
add_executable(quantizer src/main.cpp)
target_link_libraries(quantizer PRIVATE gguf_quantizer)

if(WIN32)
    set_target_properties(quantizer PROPERTIES OUTPUT_NAME "quantizer")
endif()

# --------------------------------------------------------------------------
# Avoid depending on a system-installed C/C++ runtime beyond what ships with
# the OS by default. (MSVC's static CRT is handled globally above via
# CMAKE_MSVC_RUNTIME_LIBRARY.)
# --------------------------------------------------------------------------
if(MINGW OR (UNIX AND NOT APPLE))
    # GCC-compatible runtime-linking options. Apple Clang uses the system
    # libc++/compiler runtime and rejects -static-libgcc.
    target_link_options(gguf_quantizer PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(quantizer      PRIVATE -static-libgcc -static-libstdc++)
endif()

install(TARGETS quantizer gguf_quantizer
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
)
