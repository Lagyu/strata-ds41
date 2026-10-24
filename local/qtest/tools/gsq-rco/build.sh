#!/bin/bash
# build.sh - incremental build of build-hip-mayas (the separate test build; build-hip = production is never touched) after
# fork_fmt.cuh gained IQ2_XXS / IQ2_XS as down formats: ds41_generate, native_grouped_parity, test_ds41_moe_gpu (the HIP
# expert tier vs a dequant + F32 reference on real experts) and test_ds41_moe (the CPU tier's same gate). Launch:
#   sudo systemd-run --unit=gsq-build --property=User=yuya --property=Nice=19 --property=CPUWeight=20 \
#        --property=MemoryMax=14G --collect /bin/bash /home/yuya/ds41/gsq-rco/build.sh
set -u
R=/home/yuya/llm-src/strata-ds41; B=$R/build-hip-mayas; S=/home/yuya/ds41/gsq-rco
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/yuya ROCM_PATH=/opt/rocm
cd "$R" || exit 1
exec >> "$S/build.log" 2>&1
echo "$(date '+%F %T') build.sh start"
P0=$(sha256sum build-hip/ds41_generate | cut -d' ' -f1)
nice -n 19 cmake --build "$B" --target ds41_generate native_grouped_parity test_ds41_moe_gpu test_ds41_moe -j "${JOBS:-4}" \
  || { echo "$(date '+%F %T') BUILD FAILED"; exit 1; }
ls -la "$B/ds41_generate" "$B/native_grouped_parity" "$B/test_ds41_moe_gpu" "$B/test_ds41_moe"
P1=$(sha256sum build-hip/ds41_generate | cut -d' ' -f1)
[ "$P0" = "$P1" ] && echo "production binary untouched ($P1)" || echo "WARNING: build-hip/ds41_generate changed"
echo "$(date '+%F %T') BUILD OK"
