#!/bin/bash
# chain-f1 (2026-10-21, the fused-decode binary b29c8722d + the twin back): the service-level reference, then the cheap
# exact knobs re-tuned under the new dense cost - DSpark depth 2 / 3, the twin share 0.26 / 0.30, prefetch slots 2.
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
local/sweep.sh F-ref "$B"
local/sweep.sh F-depth2 "$B" STRATA_DSPARK_DEPTH=2
local/sweep.sh F-depth3 "$B" STRATA_DSPARK_DEPTH=3
local/sweep.sh F-share26 "$B" STRATA_BLOB_MIRROR_SHARE=0.26
local/sweep.sh F-share30 "$B" STRATA_BLOB_MIRROR_SHARE=0.30
local/sweep.sh F-pf2 "$B" STRATA_D3_PF=2
(cd /home/yuya/ds41 && python3 table.py S-cap8 F-ref F-depth2 F-depth3 F-share26 F-share30 F-pf2)
# back to the default config for the service
echo "$B" > local/active-config
sudo systemctl restart strata-ds41.service
echo "chain-f1 done $(date +%H:%M)"
