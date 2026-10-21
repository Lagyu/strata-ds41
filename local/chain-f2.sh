#!/bin/bash
# chain-f2 (after chain-f1): GPU0's expert-slot count is the DMA floor's lever - the drafter pruned to its most-used
# experts (STRATA_DSPARK_KEEP, ~3 GiB back to the slots; the drafts change, the trunk's tokens do not) and a smaller
# context window (--ctx 32768 / 65536: the fp4 KV of 131072 tokens is ~3 GiB), then a second reference after the P41 move.
cd /home/yuya/llm-src/strata-ds41
until grep -q "chain-f1 done" logs/chain-f1.log 2>/dev/null; do sleep 30; done
B=local/ds41-yuya-3gpu.json
C=/home/yuya/models/dsv41-1017-strata/dspark-1017-counts.txt
local/sweep.sh F-keep96 "$B" STRATA_DSPARK_KEEP=$C:96
local/sweep.sh F-ctx32k "$B" --ctx=32768
local/sweep.sh F-ctx64k "$B" --ctx=65536
local/sweep.sh F-ref2 "$B"
(cd /home/yuya/ds41 && python3 table.py S-cap8 F-ref F-ref2 F-depth2 F-depth3 F-share26 F-share30 F-pf2 F-keep96 F-ctx32k F-ctx64k)
echo "$B" > local/active-config
sudo systemctl restart strata-ds41.service
echo "chain-f2 done $(date +%H:%M)"
