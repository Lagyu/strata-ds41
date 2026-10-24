#!/bin/bash
# build.sh - configure + build build-hip-mayas, a SEPARATE build directory (build-hip = production is never touched):
# ds41_generate with the IQ3_S down-projection kernel (src/kernels/cuda/fork/fork_fmt.cuh, for Maya-S's layers 0-3 and
# 36-39) and native_grouped_parity (the kernel test, with Maya-S's formats and shapes added). CPU only, niced, -j4 under a
# memory cap so production (109 GB locked) is never squeezed. Same cmake options as build-hip's CMakeCache.txt. Launch:
#   sudo systemd-run --unit=mayas-build --property=User=yuya --property=Nice=19 --property=CPUWeight=20 \
#        --property=MemoryMax=14G --collect /bin/bash /home/yuya/ds41/maya-s/build.sh
set -u
R=/home/yuya/llm-src/strata-ds41; B=$R/build-hip-mayas; S=/home/yuya/ds41/maya-s
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/yuya ROCM_PATH=/opt/rocm
cd "$R" || exit 1
exec >> "$S/build.log" 2>&1
echo "$(date '+%F %T') build.sh start"
sha256sum build-hip/ds41_generate | tee "$S/prod-binary.sha256"
cmake -S . -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ "-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1201" "-DGPU_TARGETS=gfx1100;gfx1201" \
  -DSTRATA_GGML_DIR="$R/third_party/llama.cpp" -DSTRATA_GGML_PREBUILT="$R/third_party/llama.cpp/build-hip" \
  -DSTRATA_DENSE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_TAB=OFF \
  -DSTRATA_MMQ_TQ_OVERLAY=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=OFF -DSTRATA_WERROR=OFF \
  -DSTRATA_PLE_FIXTURE_DIR="$R/bench/micro" || { echo "$(date '+%F %T') CONFIGURE FAILED"; exit 1; }
nice -n 19 cmake --build "$B" --target ds41_generate native_grouped_parity -j "${JOBS:-4}" || { echo "$(date '+%F %T') BUILD FAILED"; exit 1; }
ls -la "$B/ds41_generate" "$B/native_grouped_parity"
if sha256sum build-hip/ds41_generate | diff -q - "$S/prod-binary.sha256" > /dev/null; then echo "production binary untouched"; else echo "WARNING: build-hip/ds41_generate changed?!"; fi
echo "$(date '+%F %T') BUILD OK"
