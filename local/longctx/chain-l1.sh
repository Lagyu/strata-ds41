#!/bin/bash
# chain-l1 (after chain-f4): the long-context staged run.  1) NIAH prefill scaling on the default config (32K / 64K /
# 126K); 2) the 1M-context variant (local/variants/F-ctx1m.json), NIAH at 256K / 512K; 3) MRCR (openai/mrcr) rows at
# ~128K (2/4/8 needles) then ~900K (2/4/8 needles) - each 900K row is a 2-3 h prefill.  Results: local/longctx/results.jsonl,
# log: logs/chain-l1.log.  The service is left on the 1M variant at the end (restart on the default to go back).
cd /home/yuya/llm-src/strata-ds41
PY=/home/yuya/llm-src/hfenv/bin/python3
until grep -q "chain-f4b done" logs/chain-f4b.log 2>/dev/null; do sleep 30; done
until curl -s -m 5 http://127.0.0.1:8150/v1/models | grep -q loaded; do sleep 10; done
echo "== NIAH scaling, default config (ctx 131072) $(date +%H:%M)"
$PY local/longctx/niah_run.py --tokens 32768,65536,126000 --depths 0.5 --label niah-default
echo "== switching to the 1M-context variant $(date +%H:%M)"
echo local/variants/F-ctx1m.json > local/active-config
sudo systemctl restart strata-ds41.service
until curl -s -m 5 http://127.0.0.1:8150/v1/models | grep -q loaded; do sleep 10; done
grep -E "VRAM:|expert slots" local/serve-engine.log | tail -2 | cut -c1-140
$PY local/longctx/niah_run.py --tokens 262144,524288 --depths 0.5 --label niah-1m
echo "== MRCR ~128K $(date +%H:%M)"
R128=$(grep '^rows:' local/longctx/select-128k.txt | cut -d' ' -f2)
[ -n "$R128" ] && $PY local/longctx/mrcr_run.py run --rows "$R128" --label mrcr-128k
echo "== MRCR ~900K $(date +%H:%M)"
R900=$(grep '^rows:' local/longctx/select-900k.txt | cut -d' ' -f2)
[ -n "$R900" ] && $PY local/longctx/mrcr_run.py run --rows "$R900" --label mrcr-900k
echo "chain-l1 done $(date +%H:%M)"
