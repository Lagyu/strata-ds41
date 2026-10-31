#!/bin/bash
# build9.sh - the c8 candidate: the arena hint feeds the MMQ staging path by device-to-device copies; seed parts opt-in
# (ds41-fused.cu), the engine rebuilt (fused hc mixes in the span graphs), relinked as ds41_generate-c8 / (unchanged).
#   sudo systemd-run --unit=crit-build9 --collect --property=User=yuya --property=MemoryMax=12G /bin/bash /home/yuya/ds41/crit/build4.sh
set -u
W=/home/yuya/llm-src/strata-zcnvis; B=$W/build-hip-deploy; O=/home/yuya/ds41/crit/obj; P=/home/yuya/llm-src/zcnvis
export HOME=/home/yuya PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin ROCM_PATH=/opt/rocm
exec >> /home/yuya/ds41/crit/build9.log 2>&1
echo "== build9 start (arena hint d2d) $(date '+%F %T')"
nice -n 19 cmake --build "$B" --target ds41_generate -j 2 || { echo "== engine build FAILED"; exit 1; }
cd "$B" || exit 1
ninja -t commands ds41_generate | tail -1 | sed -e 's#ggml-prebuilt-fa#ggml-prebuilt-crit2#g' -e 's#-o ds41_generate #-o ds41_generate-c8 #' -e 's#link.d#link-c8.d#' > "$O/link-c8.sh"
bash "$O/link-c8.sh" || { echo "== relink c2 FAILED"; exit 1; }
ls -la ds41_generate-c8; sha256sum ds41_generate-c8 | cut -c1-16
echo "== build9 done $(date '+%F %T')"
