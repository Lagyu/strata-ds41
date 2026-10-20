#!/bin/bash
# chain-u2 (after chain-u1): the one Basalt-style lever that applies here - the conversation-adaptive arena
# (STRATA_ARENA_REBAL=1: routed-token counts, swaps of frequent file-tier experts into the arena between requests).  Its
# single-turn cost on the bench, then the multi-turn benefit with tools/turnbench.py (a 4000-token base turn, then +64,
# +182, +512 tails) on the default config and with the arena adapting.
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
# the Q8 variant again with a 4096-token prompt batch (its F16 dense half leaves too little VRAM for the 8192 one: the 4K prompt got 400)
local/sweep.sh U-q8b local/ds41-unsloth-q8.json --prompt-batch=4096
grep -E "^VRAM: |^tier: " local/serve-engine.log | tail -2 | cut -c1-160
local/sweep.sh U-rebal "$B" STRATA_ARENA_REBAL=1
grep -E "rebal|swap" local/serve-engine.log | tail -3 | cut -c1-200
# multi-turn: the service is on U-rebal now; run turnbench twice (the second run sees the adapted arena), then the default
for i in 1 2; do python3 tools/turnbench.py http://127.0.0.1:8150 --base 4000 --deltas 64,182,512 --log local/serve-engine.log 2>&1 | tail -8 | cut -c1-160; done
echo "$B" > local/active-config
sudo systemctl stop strata-ds41.service; sleep 15; sudo systemctl start strata-ds41.service
for i in 1 2; do python3 tools/turnbench.py http://127.0.0.1:8150 --base 4000 --deltas 64,182,512 --log local/serve-engine.log 2>&1 | tail -8 | cut -c1-160; done
(cd /home/yuya/ds41 && python3 table.py S-cap8 U-q8 U-q8b U-q4 U-rebal)
echo "$(date +%T) chain-u2 done"
