#!/bin/bash
# splice-chain.sh - after the B-s requantization (splice-quant.service): the full file + its check, the held-out eval
# (quality), then speed - the production config uncapped (F-ref4), B-s with the peer stores filled from the file tier
# (S-bs, the production rule) and from the arena (S-bs-arena) - then the production config back.  Each finished step
# leaves logs/splice-chain/STEP.done and a rerun skips it.  As a transient unit:
#   sudo systemd-run --unit=splice-chain --property=User=yuya --property=LimitMEMLOCK=infinity \
#     --property=LimitNOFILE=65536 --collect /bin/bash /home/yuya/llm-src/strata-ds41/local/splice-chain.sh
# Log: logs/splice-chain.log
cd /home/yuya/llm-src/strata-ds41 || exit 1
exec >> logs/splice-chain.log 2>&1
set -o pipefail
export PATH=$PATH:/opt/rocm/bin:/usr/local/bin HOME=/home/yuya
M=/mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf
B=/mnt/win/llm/splice-b
OUT=$B/DS41-1017-splice-bs.gguf
PY=/home/yuya/llm-src/hfenv/bin/python
S=logs/splice-chain
mkdir -p $S
step() { [ -e $S/$1.done ]; }
finished() { touch $S/$1.done; echo "== $1 done $(date '+%F %T')"; }
echo "== splice-chain start $(date '+%F %T')"
# the requantization unit first; then quant once more, a no-op unless an expert is missing
while systemctl is-active -q splice-quant.service; do sleep 60; done
if ! step quant; then
  $PY tools/v41/splice_b.py quant --plan $B/plan-bs.json --src $M \
    --imatrix /home/yuya/models/unsloth-1017/imatrix/imatrix_unsloth.gguf_file --qdir $B/q --threads 28 || exit 1
  finished quant
fi
if ! step write; then
  $PY tools/v41/splice_b.py write --plan $B/plan-bs.json --src $M --qdir $B/q --out $OUT | tr '\r' '\n' | tail -2 || exit 1
  $PY tools/v41/splice_b.py check --plan $B/plan-bs.json --src $M --qdir $B/q --out $OUT --samples 400 || exit 1
  finished write
fi
if ! step eval; then
  MODEL=$OUT bash local/eval-lucebox/run-eval-3gpu.sh splice-bs || exit 1
  finished eval
fi
for row in "F-ref4 local/ds41-yuya-3gpu.json" "S-bs local/ds41-yuya-3gpu-splice-bs.json" \
           "S-bs-arena local/ds41-yuya-3gpu-splice-bs.json STRATA_PEER_FILL=arena"; do
  set -- $row
  step $1 && continue
  local/sweep.sh "$@" || exit 1
  finished $1
done
(cd /home/yuya/ds41 && python3 table.py F-ref F-ref3 F-ref4 X-xq3 S-bs S-bs-arena)
echo local/ds41-yuya-3gpu.json > local/active-config
sudo systemctl restart strata-ds41.service
for i in $(seq 1 90); do curl -s -m 5 http://127.0.0.1:8150/v1/models | grep -q loaded && break; sleep 10; done
echo "== splice-chain end $(date '+%F %T') (service on local/ds41-yuya-3gpu.json)"
