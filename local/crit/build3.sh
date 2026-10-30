#!/bin/bash
# build3.sh - the c2 candidate: ggml archive crit2 = crit (matrix-core indexer) + the fused hc-tail prep / hc-mixes kernels
# (ds41-fused.cu), the engine rebuilt (fused hc mixes in the span graphs), relinked as ds41_generate-c2 / test_ds41_indexer-c2.
#   sudo systemd-run --unit=crit-build3 --collect --property=User=yuya --property=MemoryMax=12G /bin/bash /home/yuya/ds41/crit/build3.sh
set -u
W=/home/yuya/llm-src/strata-zcnvis; B=$W/build-hip-deploy; O=/home/yuya/ds41/crit/obj; P=/home/yuya/llm-src/zcnvis
export HOME=/home/yuya PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin ROCM_PATH=/opt/rocm
exec >> /home/yuya/ds41/crit/build3.log 2>&1
echo "== build3 start (skip-read) $(date '+%F %T')"
bash /home/yuya/ds41/crit/ggml-obj.sh ggml-cuda/ds41-fused.cu "$O/ds41-fused.cu.o" || { echo "== ds41-fused.cu compile FAILED"; exit 1; }
rm -rf "$P/ggml-prebuilt-crit2"; cp -r "$P/ggml-prebuilt-crit" "$P/ggml-prebuilt-crit2"
(cd "$O" && ar rs "$P/ggml-prebuilt-crit2/ggml/src/ggml-hip/libggml-hip.a" ds41-fused.cu.o) || { echo "== ar FAILED"; exit 1; }
ar p "$P/ggml-prebuilt-crit2/ggml/src/ggml-hip/libggml-hip.a" ds41-fused.cu.o | cmp - "$O/ds41-fused.cu.o" && echo "archive member swapped"
nice -n 19 cmake --build "$B" --target ds41_generate test_ds41_indexer -j 2 || { echo "== engine build FAILED"; exit 1; }
cd "$B" || exit 1
ninja -t commands ds41_generate | tail -1 | sed -e 's#ggml-prebuilt-fa#ggml-prebuilt-crit2#g' -e 's#-o ds41_generate #-o ds41_generate-c2 #' -e 's#link.d#link-c2.d#' > "$O/link-c2.sh"
bash "$O/link-c2.sh" || { echo "== relink c2 FAILED"; exit 1; }
ls -la ds41_generate-c2; sha256sum ds41_generate-c2 | cut -c1-16
echo "== build3 done $(date '+%F %T')"
