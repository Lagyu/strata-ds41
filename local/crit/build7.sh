#!/bin/bash
# build7.sh - the c6 candidate: c5 + the seed file-tier blobs read in parts by a pool over both drives (STRATA_SEED_PARTS, default 8)
# (ds41-fused.cu), the engine rebuilt (fused hc mixes in the span graphs), relinked as ds41_generate-c6 / (unchanged).
#   sudo systemd-run --unit=crit-build7 --collect --property=User=yuya --property=MemoryMax=12G /bin/bash /home/yuya/ds41/crit/build4.sh
set -u
W=/home/yuya/llm-src/strata-zcnvis; B=$W/build-hip-deploy; O=/home/yuya/ds41/crit/obj; P=/home/yuya/llm-src/zcnvis
export HOME=/home/yuya PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin ROCM_PATH=/opt/rocm
exec >> /home/yuya/ds41/crit/build7.log 2>&1
echo "== build7 start (seed parts) $(date '+%F %T')"
nice -n 19 cmake --build "$B" --target ds41_generate -j 2 || { echo "== engine build FAILED"; exit 1; }
cd "$B" || exit 1
ninja -t commands ds41_generate | tail -1 | sed -e 's#ggml-prebuilt-fa#ggml-prebuilt-crit2#g' -e 's#-o ds41_generate #-o ds41_generate-c6 #' -e 's#link.d#link-c6.d#' > "$O/link-c6.sh"
bash "$O/link-c6.sh" || { echo "== relink c2 FAILED"; exit 1; }
ls -la ds41_generate-c6; sha256sum ds41_generate-c6 | cut -c1-16
echo "== build7 done $(date '+%F %T')"
