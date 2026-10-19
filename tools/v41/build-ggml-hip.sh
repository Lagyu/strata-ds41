#!/bin/bash
# tools/v41/build-ggml-hip.sh [ARCHS] - ggml (HIP backend) from third_party/llama.cpp, llama.cpp's way, into
# third_party/llama.cpp/build-hip: the layout STRATA_GGML_PREBUILT links (ggml/src/libggml*.a, ggml/src/ggml-hip/).
# The flash-attention template instances take most of an hour per architecture on a 28-thread CPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
ARCHS=${1:-gfx1100;gfx1201}
SRC=$PWD/third_party/llama.cpp
B=$SRC/build-hip
cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ -DCMAKE_HIP_ARCHITECTURES="$ARCHS" \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_CUDA_FA=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build "$B" --target ggml ggml-base ggml-cpu ggml-hip -j "${JOBS:-24}"
ls -la "$B"/ggml/src/libggml*.a "$B"/ggml/src/ggml-hip/libggml-hip.a
echo "build-ggml-hip: done $(date +%T)"
