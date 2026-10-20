#!/bin/bash
# After chain-s2: the engine WITH the prompt-path peer helper (build-hip/ds41_generate; the sweeps before it ran the
# frozen copy), on the depth-2 configuration: 1. the helper off = the same numbers and greedy outputs as S-depth2
# (a check of the rebuilt binary); 2-3. STRATA_PEER_CHUNK=1 twice (the 4K prefill varies run to run).
# Then the held-out quality evaluation (server stopped): ppl / top-1 / KL against the FP8 1017 teacher.
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
local/sweep.sh S-d2-new "$B" DS41_EXE= STRATA_DSPARK_DEPTH=2
local/sweep.sh S-d2-pchunk "$B" DS41_EXE= STRATA_DSPARK_DEPTH=2 STRATA_PEER_CHUNK=1
local/sweep.sh S-d2-pchunk2 "$B" DS41_EXE= STRATA_DSPARK_DEPTH=2 STRATA_PEER_CHUNK=1
(cd /home/yuya/ds41 && python3 table.py S-depth2 S-d2-new S-d2-pchunk S-d2-pchunk2 && python3 bench.py --compare S-depth2 S-d2-new | head -8)
sudo systemctl stop strata-ds41.service
sleep 5
local/eval-lucebox/run-eval.sh mxfp4-abl 2>&1 | tail -12
echo "$(date +%T) chain-s3 done"
