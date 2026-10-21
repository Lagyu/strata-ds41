#!/bin/bash
# chain-x1 (2026-10-21): after chain-l3 - 1) the merged engine's parity on the production model (held-out eval must
# reproduce base-3gpu's digits: ppl 8.626284674485438 / top-1 0.9356060606060606 / KL 0.04483545063783652);
# 2) the same eval on IndrasMirror's XQ-1017 v3 (their rank file); 3) the service on the XQ v3 config + the speed
# set; the service STAYS on XQ v3 (user 2026-10-21: "use XQ v3"; local/ds41-yuya-3gpu.json = the MXFP4-ABL config
# to go back).  Log: logs/chain-x1.log.  Needs: build-hip/ds41_generate = the xolotl-merge build (logs/build-xolotl-merge.log
# "build exit 0"), /mnt/win/llm/xq-1017-v3/JOINED.ok (local/xq-join.sh).
cd /home/yuya/llm-src/strata-ds41
XQ=/mnt/win/llm/xq-1017-v3
wait_vram_zero() {
    for i in $(seq 1 90); do
        used=$(rocm-smi --showmemuse 2>/dev/null | grep -E "GPU\[0\].*VRAM%" | grep -oE "[0-9]+$")
        [ -n "$used" ] && [ "$used" -lt 3 ] && return 0
        sleep 5
    done
}
wait_loaded() { until curl -s -m 5 http://127.0.0.1:8150/v1/models | grep -q loaded; do sleep 10; done; }
until grep -q "chain-l3 done" logs/chain-l3.log 2>/dev/null; do sleep 30; done
until grep -q "build exit 0" logs/build-xolotl-merge.log 2>/dev/null; do sleep 30; done
if [ -s local/eval-lucebox/merge-abl/score.json ]; then
    echo "== 1) merged-engine parity eval: done earlier (2026-10-21 13:28, identical to base-3gpu) - skipped"
    sudo systemctl stop strata-ds41.service 2>/dev/null; wait_vram_zero
else
    echo "== 1) merged-engine parity eval on the MXFP4-ABL model $(date +%H:%M)"
    sudo systemctl stop strata-ds41.service; wait_vram_zero
    local/eval-lucebox/run-eval-3gpu.sh merge-abl 2>&1 | tail -3
    cat local/eval-lucebox/merge-abl/score.json; echo
    echo "   reference base-3gpu: $(cat local/eval-lucebox/base-3gpu/score.json)"
fi
echo "== 2) XQ v3 eval $(date +%H:%M)"
until [ -f $XQ/JOINED.ok ]; do sleep 30; done
MODEL=$XQ/V41-XQ-1017-v3.gguf local/eval-lucebox/run-eval-3gpu.sh xq3 -- --rank $XQ/rank-1017-v3.bin 2>&1 | tail -3
cat local/eval-lucebox/xq3/score.json; echo
grep -E "tier:|peer: device|VRAM:" local/eval-lucebox/xq3/seq0.log | head -4 | cut -c1-160
echo "== 3) the service on XQ v3 + the speed set $(date +%H:%M)"
local/sweep.sh X-xq3 local/ds41-yuya-3gpu-xq3.json
(cd /home/yuya/ds41 && python3 table.py F-ref F-ref3 X-xq3)
grep -E "tier:|peer: device|expert slots|DSpark:" local/serve-engine.log | tail -5 | cut -c1-160
echo local/ds41-yuya-3gpu-xq3.json > local/active-config
echo "chain-x1 done $(date +%H:%M) (service on local/ds41-yuya-3gpu-xq3.json)"
