#!/bin/bash
# chain-u1: the unsloth UD-Q8_K_XL (dense F16 from BF16) and UD-Q4_K_XL (dense Q8_0) variants - the same native MXFP4 routed
# experts as the MXFP4-ABL file, not abliterated - loaded by the engine (relaxed dense type check), benchmarked (sweep.sh), then
# scored on the held-out set (run-eval-3gpu.sh with MODEL=...).  The service goes back to the default config at the end.
cd /home/yuya/llm-src/strata-ds41
local/sweep.sh U-q8 local/ds41-unsloth-q8.json
grep -E "^loaded in|^tier: |dense requant|ds41_generate:|error|Error" local/serve-engine.log | tail -4 | cut -c1-200
local/sweep.sh U-q4 local/ds41-unsloth-q4.json
grep -E "^loaded in|^tier: |ds41_generate:|error|Error" local/serve-engine.log | tail -3 | cut -c1-200
(cd /home/yuya/ds41 && python3 table.py S-cap8 S-cap16 U-q8 U-q4)
sudo systemctl stop strata-ds41.service
sleep 20
MODEL=/mnt/win/llm/unsloth-1017/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf local/eval-lucebox/run-eval-3gpu.sh unsloth-q8
echo "eval unsloth-q8: $(cat local/eval-lucebox/unsloth-q8/score.json)"
MODEL=/mnt/win/llm/unsloth-1017-q4/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf local/eval-lucebox/run-eval-3gpu.sh unsloth-q4
echo "eval unsloth-q4: $(cat local/eval-lucebox/unsloth-q4/score.json)"
echo local/ds41-yuya-3gpu.json > local/active-config
sudo systemctl start strata-ds41.service
echo "$(date +%T) chain-u1 done"
