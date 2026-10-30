#!/bin/bash
# build5.sh - the c4 candidate: c3 + hinted reads enqueued in ascending expert order per span
# (ds41-fused.cu), the engine rebuilt (fused hc mixes in the span graphs), relinked as ds41_generate-c4 / (unchanged).
#   sudo systemd-run --unit=crit-build5 --collect --property=User=yuya --property=MemoryMax=12G /bin/bash /home/yuya/ds41/crit/build4.sh
set -u
W=/home/yuya/llm-src/strata-zcnvis; B=$W/build-hip-deploy; O=/home/yuya/ds41/crit/obj; P=/home/yuya/llm-src/zcnvis
export HOME=/home/yuya PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin ROCM_PATH=/opt/rocm
exec >> /home/yuya/ds41/crit/build5.log 2>&1
echo "== build5 start (hint order) $(date '+%F %T')"
nice -n 19 cmake --build "$B" --target ds41_generate -j 2 || { echo "== engine build FAILED"; exit 1; }
cd "$B" || exit 1
ninja -t commands ds41_generate | tail -1 | sed -e 's#ggml-prebuilt-fa#ggml-prebuilt-crit2#g' -e 's#-o ds41_generate #-o ds41_generate-c4 #' -e 's#link.d#link-c4.d#' > "$O/link-c4.sh"
bash "$O/link-c4.sh" || { echo "== relink c2 FAILED"; exit 1; }
ls -la ds41_generate-c4; sha256sum ds41_generate-c4 | cut -c1-16
echo "== build5 done $(date '+%F %T')"
