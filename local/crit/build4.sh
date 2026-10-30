#!/bin/bash
# build4.sh - the c3 candidate: c2 + the routing-hinted file-tier read stage (STRATA_CHUNK_HINT_GIB); ggml archive crit2 unchanged
# (ds41-fused.cu), the engine rebuilt (fused hc mixes in the span graphs), relinked as ds41_generate-c3 / (unchanged).
#   sudo systemd-run --unit=crit-build4 --collect --property=User=yuya --property=MemoryMax=12G /bin/bash /home/yuya/ds41/crit/build4.sh
set -u
W=/home/yuya/llm-src/strata-zcnvis; B=$W/build-hip-deploy; O=/home/yuya/ds41/crit/obj; P=/home/yuya/llm-src/zcnvis
export HOME=/home/yuya PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin ROCM_PATH=/opt/rocm
exec >> /home/yuya/ds41/crit/build4.log 2>&1
echo "== build4 start (hint stage) $(date '+%F %T')"
nice -n 19 cmake --build "$B" --target ds41_generate -j 2 || { echo "== engine build FAILED"; exit 1; }
cd "$B" || exit 1
ninja -t commands ds41_generate | tail -1 | sed -e 's#ggml-prebuilt-fa#ggml-prebuilt-crit2#g' -e 's#-o ds41_generate #-o ds41_generate-c3 #' -e 's#link.d#link-c3.d#' > "$O/link-c3.sh"
bash "$O/link-c3.sh" || { echo "== relink c2 FAILED"; exit 1; }
ls -la ds41_generate-c3; sha256sum ds41_generate-c3 | cut -c1-16
echo "== build4 done $(date '+%F %T')"
