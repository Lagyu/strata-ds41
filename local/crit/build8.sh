#!/bin/bash
# build8.sh - the c7 candidate: c6 + STRATA_CHUNK_HINT_ARENA (the spans' routed arena experts DMA'd into the prestage region under the dense half)
# (ds41-fused.cu), the engine rebuilt (fused hc mixes in the span graphs), relinked as ds41_generate-c7 / (unchanged).
#   sudo systemd-run --unit=crit-build8 --collect --property=User=yuya --property=MemoryMax=12G /bin/bash /home/yuya/ds41/crit/build4.sh
set -u
W=/home/yuya/llm-src/strata-zcnvis; B=$W/build-hip-deploy; O=/home/yuya/ds41/crit/obj; P=/home/yuya/llm-src/zcnvis
export HOME=/home/yuya PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin ROCM_PATH=/opt/rocm
exec >> /home/yuya/ds41/crit/build8.log 2>&1
echo "== build8 start (arena hint) $(date '+%F %T')"
nice -n 19 cmake --build "$B" --target ds41_generate -j 2 || { echo "== engine build FAILED"; exit 1; }
cd "$B" || exit 1
ninja -t commands ds41_generate | tail -1 | sed -e 's#ggml-prebuilt-fa#ggml-prebuilt-crit2#g' -e 's#-o ds41_generate #-o ds41_generate-c7 #' -e 's#link.d#link-c7.d#' > "$O/link-c7.sh"
bash "$O/link-c7.sh" || { echo "== relink c2 FAILED"; exit 1; }
ls -la ds41_generate-c7; sha256sum ds41_generate-c7 | cut -c1-16
echo "== build8 done $(date '+%F %T')"
