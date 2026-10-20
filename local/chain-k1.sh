#!/bin/bash
# chain-k1: the WMMA (matrix-core) flash-attention kernel for head size 512 on RDNA (ggml-cuda/fattn.cu guard relaxed) vs the
# tile kernel: the serve bench + turnbench on the new binary (build-hip/ds41_generate) and on the frozen tile-kernel binary
# (build-hip/ds41_generate.exact), then the held-out eval on the new binary (accumulation order changes; quality must not).
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
local/sweep.sh S-wmma "$B"
python3 tools/turnbench.py http://127.0.0.1:8150 --base 4000 --deltas 64,182,512 --log local/serve-engine.log 2>&1 | grep -E "^turn" | cut -c1-150
local/sweep.sh S-tile "$B" DS41_EXE=/home/yuya/llm-src/strata-ds41/build-hip/ds41_generate.exact
python3 tools/turnbench.py http://127.0.0.1:8150 --base 4000 --deltas 64,182,512 --log local/serve-engine.log 2>&1 | grep -E "^turn" | cut -c1-150
(cd /home/yuya/ds41 && python3 table.py S-cap8 S-cap16 S-tile S-wmma && python3 bench.py --compare S-tile S-wmma | head -6)
grep -E "^serve: prompt phases" local/serve-engine.log | tail -4 | cut -c1-200
sudo systemctl stop strata-ds41.service; sleep 20
local/eval-lucebox/run-eval-3gpu.sh wmma
echo "eval wmma: $(cat local/eval-lucebox/wmma/score.json)"
echo "$B" > local/active-config
sudo systemctl start strata-ds41.service
echo "$(date +%T) chain-k1 done"
