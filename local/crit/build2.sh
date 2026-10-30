#!/bin/bash
# build2.sh - rebuild ds41_generate + test_ds41_indexer of the critpath worktree, then relink both against the swapped ggml
# archive (ggml-prebuilt-crit = production's archives with the matrix-core indexer member) as *-wmma.
#   sudo systemd-run --unit=crit-build2 --collect --property=User=yuya --property=MemoryMax=12G /bin/bash /home/yuya/ds41/crit/build2.sh
set -u
W=/home/yuya/llm-src/strata-zcnvis; B=$W/build-hip-deploy; O=/home/yuya/ds41/crit/obj
export HOME=/home/yuya PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin ROCM_PATH=/opt/rocm
exec >> /home/yuya/ds41/crit/build2.log 2>&1
echo "== build2 start $(date '+%F %T') head=$(git -C $W log -1 --format=%h) $(git -C $W status --short | wc -l) changed files"
cmake --build "$B" --target ds41_generate test_ds41_indexer -j 3 || { echo "== build FAILED $(date +%T)"; exit 1; }
cd "$B" || exit 1
ninja -t commands ds41_generate | tail -1 | sed -e 's#ggml-prebuilt-fa#ggml-prebuilt-crit#g' -e 's#-o ds41_generate #-o ds41_generate-wmma #' -e 's#link.d#link-wmma.d#' > "$O/link-wmma.sh"
bash "$O/link-wmma.sh" || { echo "== relink ds41_generate-wmma FAILED"; exit 1; }
ninja -t commands test_ds41_indexer | tail -1 | sed -e 's#ggml-prebuilt-fa#ggml-prebuilt-crit#g' -e 's#-o test_ds41_indexer #-o test_ds41_indexer-wmma #' -e 's#link.d#link-tw.d#' > "$O/link-test.sh"
bash "$O/link-test.sh" || { echo "== relink test FAILED"; exit 1; }
ls -la ds41_generate ds41_generate-wmma test_ds41_indexer-wmma
sha256sum ds41_generate-wmma | cut -c1-16
echo "== build2 done $(date '+%F %T')"
