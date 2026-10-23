#!/bin/bash
# chain-l4 (2026-10-21, after chain-l3): the clean rerun of the sweep rows that the storage session's moves made noisy
# (its I/O ended 10:31) - the reference, DSpark keep 96 and ctx 32K on the default 3-GPU config - then the default
# config back.  Log: logs/chain-l4.log; rows in /home/yuya/ds41/results.jsonl (table.py).
cd /home/yuya/llm-src/strata-ds41
until grep -q "chain-l3 done" logs/chain-l3.log 2>/dev/null; do sleep 30; done
until curl -s -m 5 http://127.0.0.1:8150/v1/models | grep -q loaded; do sleep 10; done
B=local/ds41-yuya-3gpu.json
echo "== clean sweeps $(date +%H:%M)"
local/sweep.sh F-ref4 "$B"
local/sweep.sh F-keep96c "$B" STRATA_DSPARK_KEEP=/home/yuya/models/dsv41-1017-strata/dspark-1017-counts.txt:96
local/sweep.sh F-ctx32k2 "$B" --ctx=32768
(cd /home/yuya/ds41 && python3 table.py F-ref F-ref3 F-ref4 F-keep96 F-keep96b F-keep96c F-ctx32k F-ctx32k2)
echo "$B" > local/active-config
sudo systemctl restart strata-ds41.service
until curl -s -m 5 http://127.0.0.1:8150/v1/models | grep -q loaded; do sleep 10; done
echo "chain-l4 done $(date +%H:%M)"
