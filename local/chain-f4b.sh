#!/bin/bash
# chain-f4b: the drafter-on-a-peer variants and the quiet reference again, on the fixed binary (chain-f4's runs
# failed: the STRATA_DSPARK_DEVICE edit had dropped the sidecar / trunk paths and max_append).
cd /home/yuya/llm-src/strata-ds41
V=/tmp/claude-1000/-/400248c7-d38c-5cd1-a998-fe708c2794b7/tasks/b97006747.output
until [ -f "$V" ] && grep -q "^[0-9][0-9]:[0-9][0-9]$" "$V" 2>/dev/null; do sleep 15; done   # the verification request finished
B=local/ds41-yuya-3gpu.json
local/sweep.sh F-dspdev1 "$B" STRATA_DSPARK_DEVICE=1 STRATA_PEER_GIB=22,20
local/sweep.sh F-dspdev2 "$B" STRATA_DSPARK_DEVICE=2 STRATA_PEER_GIB=28,14
local/sweep.sh F-ref3 "$B"
local/sweep.sh F-keep96b "$B" STRATA_DSPARK_KEEP=/home/yuya/models/dsv41-1017-strata/dspark-1017-counts.txt:96
(cd /home/yuya/ds41 && python3 table.py S-cap8 F-ref F-ref3 F-keep96 F-keep96b F-dspdev1 F-dspdev2)
grep -E "DSpark: on|expert_slots" local/serve-engine.log | tail -3 | cut -c1-160
echo "$B" > local/active-config
sudo systemctl restart strata-ds41.service
echo "chain-f4b done $(date +%H:%M)"
