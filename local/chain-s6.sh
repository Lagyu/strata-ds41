#!/bin/bash
# After the profile run: the decode-side knobs on the default config (depth 1): Lucebox's bigger MXFP4-Q8 drafter
# (7.4 GiB sidecar; Lucebox's 0.94 code acceptance was with it), n-gram lookup drafts (STRATA_SERVE_LOOKUP, exact:
# verified by the trunk), the confidence cut (STRATA_DSPARK_CONF=0.6: only the leading drafts the head trusts are
# verified), and the file-tier reader count (STRATA_D3_READERS 16 / 64 vs 32).
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
# the binary from before the MXFP4 multi-entry kernels and the blob twin (the engine reads DS41_EXE from the config's env)
X=DS41_EXE=/home/yuya/llm-src/strata-ds41/build-hip/ds41_generate.sweep
local/sweep.sh S-ref1 "$B" $X
local/sweep.sh S-mxdraft "$B" $X STRATA_SERVE_DSPARK=/home/yuya/models/dsv41-1017-strata/DSpark-1017-MXFP4-Q8-sidecar.gguf
local/sweep.sh S-lookup14 "$B" $X STRATA_SERVE_LOOKUP=14
local/sweep.sh S-conf06 "$B" $X STRATA_DSPARK_CONF=0.6
local/sweep.sh S-readers16 "$B" $X STRATA_D3_READERS=16
local/sweep.sh S-readers64 "$B" $X STRATA_D3_READERS=64
(cd /home/yuya/ds41 && python3 table.py S-d1 S-prof S-ref1 S-mxdraft S-lookup14 S-conf06 S-readers16 S-readers64)
echo "$(date +%T) chain-s6 done"
