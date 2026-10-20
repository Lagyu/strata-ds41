#!/bin/bash
# After chain-s7: the batched peer-row deposit (one H2D + rows_scatter per layer instead of one cudaMemcpyAsync per
# peer row) on the default config, plain and profiled.
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
local/sweep.sh S-deposit "$B"
local/sweep.sh S-deposit-prof "$B" DS41_PROF_X=1
(cd /home/yuya/ds41 && python3 table.py S-ref1 S-new S-twin22 S-twin30 S-new-prof S-deposit S-deposit-prof && python3 bench.py --compare S-new S-deposit | head -6)
grep "\[prof_x\] [0-9]* calls" local/serve-engine.log | tail -2 | cut -c1-230
echo "$(date +%T) chain-s8 done"
