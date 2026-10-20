#!/bin/bash
# After chain-s6: the NEW binary (MXFP4 on the multi-entry kernels: an expert's rows read once per pass, not once per
# routed row; STRATA_BLOB_TWIN: the model file's copy on the Linux SSD serves a share of every file-tier blob's parts).
# 1. the default config on the new binary (the greedy outputs must match S-d1's: the kernels claim bit-identical rows);
# 2-3. the twin at 0.22 and 0.30 of each blob's parts (the 980 reads 1.3 GB/s direct, the P41 ~5).
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
T=STRATA_BLOB_TWIN=/home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf
local/sweep.sh S-new "$B"
local/sweep.sh S-twin22 "$B" $T STRATA_BLOB_MIRROR_SHARE=0.22
local/sweep.sh S-twin30 "$B" $T STRATA_BLOB_MIRROR_SHARE=0.30
# the profile again on the new binary (the frozen one: launch 2.6 ms + sum+sync 3.4 ms per layer-call, 10.9 distinct
# experts for 36 routed rows; STRATA_TIER_GRAPH covers only the one-token path, not the DSpark passes)
local/sweep.sh S-new-prof "$B" DS41_PROF_X=1
(cd /home/yuya/ds41 && python3 table.py S-d1 S-ref1 S-new S-twin22 S-twin30 S-new-prof && python3 bench.py --compare S-d1 S-new | head -6)
echo "$(date +%T) chain-s7 done"
