#!/bin/bash
# After chain-s3: on the new binary at depth 2 (the best so far): DSpark depth 1; the bit-exact 4-bit KV cache
# (--kv-fp4, Xolotl: V4.1 rounds its compressed KV to that grid anyway) in place of q8; a 32K context (the KV
# reservation shrinks, the VRAM expert tier grows); STRATA_SERVE_STEP_MAX=32 (prompts above 32 tokens take the
# layer-major batch instead of 16-row passes: the short prompts prefill at 3-6 tok/s on the passes).
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
local/sweep.sh S-d1 "$B" DS41_EXE= STRATA_DSPARK_DEPTH=1
local/sweep.sh S-d2-kvfp4 "$B" DS41_EXE= STRATA_DSPARK_DEPTH=2 --kv-q8= --kv-fp4=on
local/sweep.sh S-d2-ctx32 "$B" DS41_EXE= STRATA_DSPARK_DEPTH=2 --ctx=32768
local/sweep.sh S-d2-step32 "$B" DS41_EXE= STRATA_DSPARK_DEPTH=2 STRATA_SERVE_STEP_MAX=32
(cd /home/yuya/ds41 && python3 table.py S-depth2 S-d2-new S-d2-pchunk S-d2-pchunk2 S-d1 S-d2-kvfp4 S-d2-ctx32 S-d2-step32)
echo "$(date +%T) chain-s4 done"
