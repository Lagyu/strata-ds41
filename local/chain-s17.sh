#!/bin/bash
# chain-s17 (binary with: Engram rows of a pass read asynchronously (STRATA_ENG_SYNC=1 = old), the GPU-timeline profile,
# --skip-file in the pass + chunk paths, the file penalty peer-aware + STRATA_BIAS_NOW): the plain new binary, its
# timeline profile, then the quality-trading levers' SPEED (skip-file 0.10/0.15/0.20, route-bias-file 0.2/0.4) and
# their held-out QUALITY on the production residency (run-eval-3gpu.sh, server stopped).
cd /home/yuya/llm-src/strata-ds41
B=local/ds41-yuya-3gpu.json
local/sweep.sh S-old "$B" STRATA_ENG_SYNC=1 STRATA_D3_FG_PIPE=0
local/sweep.sh S-eng "$B" STRATA_D3_FG_PIPE=0
local/sweep.sh S-fgpipe "$B"
local/sweep.sh S-fgpipe-r16 "$B" STRATA_D3_READERS=16
local/sweep.sh S-fgpipe-r8 "$B" STRATA_D3_READERS=8
local/sweep.sh S-prof5 "$B" DS41_PROF_X=1
grep -E "\[prof_x\] (timeline|[0-9]+ calls)" local/serve-engine.log | tail -2 | cut -c1-420
local/sweep.sh S-skip10 "$B" --skip-file=0.10
local/sweep.sh S-skip15 "$B" --skip-file=0.15
local/sweep.sh S-skip20 "$B" --skip-file=0.20
local/sweep.sh S-rbf02 "$B" --route-bias-file=0.2
local/sweep.sh S-rbf04 "$B" --route-bias-file=0.4
local/sweep.sh S-thp "$B" STRATA_NO_LARGEPAGES=
grep -E "^arena: pinned" local/serve-engine.log | tail -1 | cut -c1-160
local/sweep.sh S-d2c70 "$B" STRATA_DSPARK_DEPTH=2 STRATA_DSPARK_CONF=0.7
local/sweep.sh S-d2c85 "$B" STRATA_DSPARK_DEPTH=2 STRATA_DSPARK_CONF=0.85
grep -E "^serve: [0-9]+ tokens in" local/serve-engine.log | tail -24 | cut -c1-200
(cd /home/yuya/ds41 && python3 table.py S-reorder S-prof4 S-old S-eng S-fgpipe S-fgpipe-r16 S-fgpipe-r8 S-prof5 S-skip10 S-skip15 S-skip20 S-rbf02 S-rbf04 S-thp S-d2c70 S-d2c85)
sudo systemctl stop strata-ds41.service
sleep 20
for tag in base-3gpu skip10 skip15 skip20 rbf02 rbf04; do
  case $tag in
    base-3gpu) local/eval-lucebox/run-eval-3gpu.sh $tag ;;
    skip10) local/eval-lucebox/run-eval-3gpu.sh $tag -- --skip-file 0.10 ;;
    skip15) local/eval-lucebox/run-eval-3gpu.sh $tag -- --skip-file 0.15 ;;
    skip20) local/eval-lucebox/run-eval-3gpu.sh $tag -- --skip-file 0.20 ;;
    rbf02) local/eval-lucebox/run-eval-3gpu.sh $tag STRATA_BIAS_NOW=1 -- --route-bias-file 0.2 ;;
    rbf04) local/eval-lucebox/run-eval-3gpu.sh $tag STRATA_BIAS_NOW=1 -- --route-bias-file 0.4 ;;
  esac
  echo "eval $tag: $(cat local/eval-lucebox/$tag/score.json)"
done
echo "$B" > local/active-config
sudo systemctl start strata-ds41.service
echo "$(date +%T) chain-s17 done"
