#!/bin/bash
# build.sh - build ds41_generate of the critpath worktree (strata-zcnvis, branch critpath) in build-hip-deploy (prebuilt ggml-prebuilt-fa).
#   sudo systemd-run --unit=crit-build --collect --property=User=yuya --property=MemoryMax=12G /bin/bash /home/yuya/ds41/crit/build.sh
set -u
W=/home/yuya/llm-src/strata-zcnvis
export HOME=/home/yuya PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin ROCM_PATH=/opt/rocm
exec >> /home/yuya/ds41/crit/build.log 2>&1
echo "== build start $(date '+%F %T') head=$(git -C $W log -1 --format=%h) $(git -C $W status --short | wc -l) changed files"
cmake --build "$W/build-hip-deploy" --target ds41_generate -j 3 || { echo "== build FAILED $(date +%T)"; exit 1; }
ls -la "$W/build-hip-deploy/ds41_generate"
sha256sum "$W/build-hip-deploy/ds41_generate" | cut -c1-16
echo "== build done $(date '+%F %T')"
