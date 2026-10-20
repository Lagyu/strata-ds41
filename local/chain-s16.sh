#!/bin/bash
# chain-s16 (new binary 13:57: DS41_PROF_X splits sum+sync into the card's backlog and the peers' wait; STRATA_D3_FILE_PRED
# measures the pass-path prediction hit rates for the file-tier experts): the profile, the prediction measurement, then
# exact env levers - STRATA_LRU_WIN=16 (VRAM admission window), STRATA_D3_PARTS=8 (2.2 MB read parts), STRATA_D3_PF=2.
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
local/sweep.sh S-prof4 "$B" DS41_PROF_X=1 STRATA_ENG_PROF=1
grep -E "\[prof_x\] [0-9]+ calls|engram decode reads" local/serve-engine.log | tail -3 | cut -c1-330
local/sweep.sh S-fpred "$B" STRATA_D3_FILE_PRED=1
grep -E "\[fpred\]" local/serve-engine.log | tail -2 | cut -c1-400
local/sweep.sh S-lruwin16 "$B" STRATA_LRU_WIN=16
local/sweep.sh S-parts8 "$B" STRATA_D3_PARTS=8
local/sweep.sh S-d3pf2 "$B" STRATA_D3_PF=2
(cd /home/yuya/ds41 && python3 table.py S-reorder S-reorder-prof S-prof4 S-fpred S-lruwin16 S-parts8 S-d3pf2)
echo "$B" > local/active-config
sudo systemctl restart strata-ds41.service
echo "$(date +%T) chain-s16 done"
