#!/bin/bash
# local/secprof/chain-sec.sh - the security-aware residency splice end to end; every finished step leaves
# logs/chain-sec/STEP.done and a rerun skips it:
#   ref    production's logits on the held-out security set (run-eval-sec.sh prod) = the reference
#   plan   rank-sec.bin = the security profile 0.5 + Lucebox 0.25 + unsloth 0.25; plan-sec.json (B-s's class sizes and
#          formats); the requantization restarted on it (the stores keep every expert already done)
#   quant  waits for it
#   write  DS41-1017-splice-sec.gguf + check
#   eval   generic held-out (run-eval-3gpu.sh splice-sec) + security held-out (run-eval-sec.sh splice-sec)
#   sweeps F-ref4 (production, uncapped) + side-by-side tasks / S-sec + side-by-side tasks / S-sec-arena
# then the production config back with the service running.  As a transient unit:
#   sudo systemd-run --unit=chain-sec --property=User=yuya --property=LimitMEMLOCK=infinity \
#     --property=LimitNOFILE=65536 --collect /bin/bash /home/yuya/llm-src/strata-ds41/local/secprof/chain-sec.sh
# Log: logs/chain-sec.log
cd /home/yuya/llm-src/strata-ds41 || exit 1
exec >> logs/chain-sec.log 2>&1
set -o pipefail
export PATH=$PATH:/opt/rocm/bin:/usr/local/bin HOME=/home/yuya
M=/mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf
B=/mnt/win/llm/splice-b
S=/home/yuya/datasets/security-profile/build
IM=/home/yuya/models/unsloth-1017/imatrix/imatrix_unsloth.gguf_file
OUT=$B/DS41-1017-splice-sec.gguf
PY=/home/yuya/llm-src/hfenv/bin/python
D=logs/chain-sec
mkdir -p $D
step() { [ -e $D/$1.done ]; }
finished() { touch $D/$1.done; echo "== $1 done $(date '+%F %T')"; }
echo "== chain-sec start $(date '+%F %T')"
while systemctl is-active -q secprof.service; do sleep 60; done
[ -s $S/route-counts.final.csv ] || { echo "no security profile ($S/route-counts.final.csv)"; exit 1; }
if ! step ref; then
  bash local/secprof/run-eval-sec.sh prod || exit 1
  finished ref
fi
if ! step plan; then
  $PY tools/v41/splice_b.py rank --csv $S/route-counts.final.csv --csv /home/yuya/ds41/routing-stats.csv --imatrix $IM \
    --weights 0.5,0.25,0.25 --out $B/rank-sec.bin || exit 1
  $PY tools/v41/splice_b.py plan --rank $B/rank-sec.bin --hot 2741 --warm 7540 --warm-type iq3_s,q3_k --cold-type iq2_s \
    --imatrix $IM --out $B/plan-sec.json || exit 1
  sudo systemctl stop splice-quant.service
  sleep 5
  sudo systemd-run --unit=splice-quant --property=User=yuya --collect /bin/bash $PWD/local/splice-quant.sh $B/plan-sec.json || exit 1
  finished plan
fi
sleep 30
while systemctl is-active -q splice-quant.service; do sleep 60; done
if ! step quant; then
  $PY tools/v41/splice_b.py quant --plan $B/plan-sec.json --src $M --imatrix $IM --qdir $B/q --threads 28 || exit 1
  finished quant
fi
if ! step write; then
  $PY tools/v41/splice_b.py write --plan $B/plan-sec.json --src $M --qdir $B/q --out $OUT | tr '\r' '\n' | tail -2 || exit 1
  $PY tools/v41/splice_b.py check --plan $B/plan-sec.json --src $M --qdir $B/q --out $OUT --samples 400 || exit 1
  finished write
fi
if ! step eval; then
  MODEL=$OUT bash local/eval-lucebox/run-eval-3gpu.sh splice-sec || exit 1
  RANK=$B/rank-sec.bin bash local/secprof/run-eval-sec.sh splice-sec $OUT || exit 1
  finished eval
fi
for row in "F-ref4 local/ds41-yuya-3gpu.json" "S-sec local/ds41-yuya-3gpu-splice-sec.json" \
           "S-sec-arena local/ds41-yuya-3gpu-splice-sec.json STRATA_PEER_FILL=arena"; do
  set -- $row
  step $1 && continue
  local/sweep.sh "$@" || exit 1
  case $1 in
    F-ref4) $PY local/secprof/side.py prod --tasks 6 --max-tokens 2000 ;;
    S-sec) $PY local/secprof/side.py splice-sec --tasks 6 --max-tokens 2000 ;;
  esac
  finished $1
done
(cd /home/yuya/ds41 && python3 table.py F-ref F-ref3 F-ref4 X-xq3 S-sec S-sec-arena)
echo local/ds41-yuya-3gpu.json > local/active-config
sudo systemctl restart strata-ds41.service
for i in $(seq 1 90); do curl -s -m 5 http://127.0.0.1:8150/v1/models | grep -q loaded && break; sleep 10; done
echo "== chain-sec end $(date '+%F %T') (service on local/ds41-yuya-3gpu.json)"
