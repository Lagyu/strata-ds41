#!/bin/bash
# After the strata-3gpu bench: the first sweep on the three-GPU configuration (thinking off, as the Lucebox rows).
# 1. the same config as a thinking-off reference; 2-3. DSpark depth 3 / 2 (fewer rows per verify pass -> fewer
# distinct experts per pass; Lucebox's best was q=3); 4-5. the batched PCIe share 0.7 / 0.5 (the CPU pool takes
# the rest of the arena misses: "0 cpu" so far, 12 threads idle); 6. 20 pool threads (all physical cores).
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
local/sweep.sh S-3gpu-ref "$B"
local/sweep.sh S-depth3 "$B" STRATA_DSPARK_DEPTH=3
local/sweep.sh S-depth2 "$B" STRATA_DSPARK_DEPTH=2
local/sweep.sh S-pcie07 "$B" STRATA_TIER_BATCH_PCIE=0.7
local/sweep.sh S-pcie05 "$B" STRATA_TIER_BATCH_PCIE=0.5
local/sweep.sh S-thr20 "$B" --threads=20
# 7. the arena on 2 MiB pages (the fork's configs skip them: STRATA_NO_LARGEPAGES=1): fewer DMA descriptors per blob
local/sweep.sh S-lpages "$B" STRATA_NO_LARGEPAGES=
(cd /home/yuya/ds41 && python3 table.py S-3gpu-ref S-depth3 S-depth2 S-pcie07 S-pcie05 S-thr20 S-lpages)
echo "$(date +%T) chain-s1 done"
