#!/bin/bash
# chain-s18 (after chain-s17; binary with the per-call reader cap: STRATA_D3_READERS_PER_BLOB=8 default): the cap itself
# (plain = 8 per blob, 4, 16, and 32 readers = the uncapped old behavior), then the predicted prefetch's slot count
# (STRATA_D3_PF=2 gave 4K decode 6.3 -> 7.0 in chain-s16) on top of the new default.
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
local/sweep.sh S-cap8 "$B"
local/sweep.sh S-cap4 "$B" STRATA_D3_READERS_PER_BLOB=4
local/sweep.sh S-cap16 "$B" STRATA_D3_READERS_PER_BLOB=16
local/sweep.sh S-cap32 "$B" STRATA_D3_READERS_PER_BLOB=32
local/sweep.sh S-pf2 "$B" STRATA_D3_PF=2
local/sweep.sh S-pf3 "$B" STRATA_D3_PF=3
local/sweep.sh S-pf4 "$B" STRATA_D3_PF=4
local/sweep.sh S-pf2b "$B" STRATA_D3_PF=2
# the drafter pruned to its most-used experts (STRATA_DSPARK_KEEP): ~3 GiB of VRAM back to the trunk's expert cache; the
# counts come from a run with STRATA_DSPARK_COUNT (written when the engine exits)
C=/home/yuya/models/dsv41-1017-strata/dspark-1017-counts.txt
local/sweep.sh S-count "$B" STRATA_DSPARK_COUNT=$C
sleep 5; ls -la $C 2>&1 | cut -c1-120
if [ -s "$C" ]; then
  local/sweep.sh S-keep96 "$B" STRATA_DSPARK_KEEP=$C:96
  local/sweep.sh S-keep128 "$B" STRATA_DSPARK_KEEP=$C:128,96,96
  grep -E "^DSpark: .*GB of weights|^VRAM: " local/serve-engine.log | tail -2 | cut -c1-160
fi
(cd /home/yuya/ds41 && python3 table.py S-old S-eng S-fgpipe S-fgpipe-r8 S-cap8 S-cap4 S-cap16 S-cap32 S-pf2 S-pf3 S-pf4 S-pf2b S-count S-keep96 S-keep128)
echo "$B" > local/active-config
sudo systemctl restart strata-ds41.service
echo "$(date +%T) chain-s18 done"
