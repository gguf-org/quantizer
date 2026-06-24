# 3.15+ is required so that CMP0091 defaults to NEW, which makes the
# MSVC_RUNTIME_LIBRARY / CMAKE_MSVC_RUNTIME_LIBRARY settings below actually
# take effect (on 3.14 they are silently ignored and MSVC falls back to the
# dynamic CRT baked into CMAKE_<LANG>_FLAGS_<CONFIG>).
cmake_minimum_required(VERSION 3.15)
project(gguf-quantizer C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# --------------------------------------------------------------------------
# Vendor ggml (gguf I/O + quantization kernels)
# checkout. We only need "ggml-base" (ggml.c/ggml-quants.c/gguf.cpp) - no
# compute backends, no OpenMP, no examples/tests - to keep the dependency
# surface (and the resulting DLL set) as small as possible.
# --------------------------------------------------------------------------
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml" CACHE PATH "path to ggml")

set(GGML_CPU OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

if(NOT DEFINED GGML_NATIVE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
endif()
if(NOT GGML_NATIVE)
    # portable baseline (anything from ~2013 onward) instead of -march=native,
    # so the resulting binaries run on any x86_64 machine.
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
    set(GGML_FMA ON CACHE BOOL "" FORCE)
    set(GGML_F16C ON CACHE BOOL "" FORCE)
endif()

if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()

# --------------------------------------------------------------------------
# Make the build self-contained so the contents of bin/ can be copied to a
# machine that never had a compiler/redistributable installed:
#  - MSVC: statically link the CRT (no MSVCP140.dll/VCRUNTIME140*.dll/
#    api-ms-win-crt-*.dll dependency). Must be set before add_subdirectory()
#    so ggml's own targets (ggml.dll, ggml-base.dll) inherit it too, not just
#    gguf_quantizer/quantizer.
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

add_subdirectory(${GGML_DIR} ${CMAKE_BINARY_DIR}/ggml)

# --------------------------------------------------------------------------
# gguf_quantizer: shared library with the reusable quantization logic
# --------------------------------------------------------------------------
add_library(gguf_quantizer SHARED src/quantize.cpp src/quantize.h)
target_include_directories(gguf_quantizer PUBLIC src PRIVATE ${GGML_DIR}/include)
target_link_libraries(gguf_quantizer PRIVATE ggml-base)
target_compile_definitions(gguf_quantizer PRIVATE GGUF_QUANTIZER_BUILD PUBLIC GGUF_QUANTIZER_SHARED)

find_package(Threads REQUIRED)
target_link_libraries(gguf_quantizer PRIVATE Threads::Threads)

set_target_properties(gguf_quantizer PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# --------------------------------------------------------------------------
# quantizer: CLI front-end
# --------------------------------------------------------------------------
add_executable(quantizer src/main.cpp)
target_link_libraries(quantizer PRIVATE gguf_quantizer)

if(WIN32)
    set_target_properties(quantizer PROPERTIES OUTPUT_NAME "quantizer")
endif()

# --------------------------------------------------------------------------
# Make the build self-contained: avoid depending on a system-installed
# C/C++ runtime beyond what ships with the OS by default. (MSVC's static CRT
# is handled globally above via CMAKE_MSVC_RUNTIME_LIBRARY, before
# add_subdirectory(ggml), so ggml's own DLLs get it too.)
# --------------------------------------------------------------------------
if(MINGW)
    target_link_options(gguf_quantizer PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(quantizer      PRIVATE -static-libgcc -static-libstdc++)
elseif(UNIX AND NOT APPLE)
    # These are GCC-compatible runtime-linking options. Apple Clang uses the
    # system libc++/compiler runtime and rejects -static-libgcc.
    target_link_options(gguf_quantizer PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(quantizer      PRIVATE -static-libgcc -static-libstdc++)
endif()

install(TARGETS quantizer gguf_quantizer
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
)
