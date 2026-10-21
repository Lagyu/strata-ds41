#!/bin/bash
# chain-l3 (2026-10-21, after the 512K needle OOM'd the prompt batch and left the service with 4 GiB of VRAM free):
# restart on the rebuilt engine (context-aware prompt-batch sizing) + the revised 1M variant (drafter on the R9700),
# rerun 1) the MRCR ~128K rows (2/4/8 needles, the tooling check) and 2) the 512K needle (the VRAM fix's validation),
# then 3) restore the default config.  The 1M rows stay on hold (select-900k.txt's rows line is empty).
# Log: logs/chain-l3.log.  Results: local/longctx/results.jsonl.
cd /home/yuya/llm-src/strata-ds41
PY=/home/yuya/llm-src/hfenv/bin/python3
wait_vram_zero() {   # the old engine must have released its VRAM before the new one starts (else the new one OOMs)
    for i in $(seq 1 60); do
        used=$(rocm-smi --showmemuse 2>/dev/null | grep -E "GPU\[0\].*Used" | grep -oE "[0-9]+" | tail -1)
        [ -n "$used" ] && [ "$used" -lt 3 ] && return 0
        sleep 5
    done
}
wait_loaded() { until curl -s -m 5 http://127.0.0.1:8150/v1/models | grep -q loaded; do sleep 10; done; }
echo "== restart on the 1M variant, rebuilt engine $(date +%H:%M)"
sudo systemctl stop strata-ds41.service; wait_vram_zero
echo local/variants/F-ctx1m.json > local/active-config
sudo systemctl start strata-ds41.service; wait_loaded
grep -E "VRAM:|expert slots|DSpark:" local/serve-engine.log | tail -3 | cut -c1-160
echo "== MRCR ~128K $(date +%H:%M)"
R128=$(grep '^rows:' local/longctx/select-128k.txt | cut -d' ' -f2)
[ -n "$R128" ] && $PY local/longctx/mrcr_run.py run --rows "$R128" --label mrcr-128k
echo "== 512K needle (VRAM fix validation) $(date +%H:%M)"
$PY local/longctx/niah_run.py --tokens 524288 --depths 0.5 --label niah-1m-fix
grep -E "prompt batch|graph buffer|prompt phases: 5242" local/serve-engine.log | tail -4 | cut -c1-160
echo "== restore the default config $(date +%H:%M)"
sudo systemctl stop strata-ds41.service; wait_vram_zero
echo local/ds41-yuya-3gpu.json > local/active-config
sudo systemctl start strata-ds41.service; wait_loaded
echo "chain-l3 done $(date +%H:%M)"
