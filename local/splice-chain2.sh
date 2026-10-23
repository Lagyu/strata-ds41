#!/bin/bash
# splice-chain2.sh - the B-s splice (3.29 bpw, stores in /mnt/win/llm/splice-b/q since 2026-10-22 02:48): write the full
# file + check it, the held-out eval (ppl / top-1 / KL vs the FP8 teacher, the same gate as base-3gpu), then speed on a
# splice config derived from the CURRENT production config (same 1M layout; model + rank swapped, no blob twin): F-ref5
# (production, same night), S-bs, S-bs-arena.  The service is stopped first (the write must not compete with the 100 GB
# engine for RAM) and started again on the production config at the end.  Each finished step leaves
# logs/splice-chain2/STEP.done and a rerun skips it.  As a transient unit:
#   sudo systemd-run --unit=splice-chain2 --property=User=yuya --property=LimitMEMLOCK=infinity \
#     --property=LimitNOFILE=65536 --collect /bin/bash /home/yuya/llm-src/strata-ds41/local/splice-chain2.sh
cd /home/yuya/llm-src/strata-ds41 || exit 1
exec >> logs/splice-chain2.log 2>&1
set -o pipefail
export PATH=$PATH:/opt/rocm/bin:/usr/local/bin HOME=/home/yuya
M=/mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf
B=/mnt/win/llm/splice-b
OUT=$B/DS41-1017-splice-bs.gguf
PY=/home/yuya/llm-src/hfenv/bin/python
S=logs/splice-chain2
mkdir -p $S
step() { [ -e $S/$1.done ]; }
finished() { touch $S/$1.done; echo "== $1 done $(date '+%F %T')"; }
wait_vram() {
  for i in $(seq 1 90); do
    busy=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk -F': ' '{s+=$NF} END{print s+0}')
    [ "$busy" = "0" ] && break
    sleep 2
  done
}
echo "== splice-chain2 start $(date '+%F %T')"
sudo systemctl stop strata-ds41.service
wait_vram
if ! step write; then
  echo "== write start $(date '+%F %T')"
  $PY tools/v41/splice_b.py write --plan $B/plan-bs.json --src $M --qdir $B/q --out $OUT | tr '\r' '\n' | tail -3 || exit 1
  ls -la $OUT
  $PY tools/v41/splice_b.py check --plan $B/plan-bs.json --src $M --qdir $B/q --out $OUT --samples 400 || exit 1
  finished write
fi
if ! step eval; then
  echo "== eval start $(date '+%F %T')"
  MODEL=$OUT bash local/eval-lucebox/run-eval-3gpu.sh splice-bs || exit 1
  echo "== reference (base-3gpu):"; cat local/eval-lucebox/base-3gpu/score.json
  finished eval
fi
# the splice config from the current production config: model + rank swapped, the blob twin (a copy of the MXFP4 file)
# and the production route counts dropped
python3 - <<'PY'
import json, collections
p = json.load(open('local/ds41-yuya-3gpu.json'), object_pairs_hook=collections.OrderedDict)
a = p['args']
a[a.index('-m') + 1] = '/mnt/win/llm/splice-b/DS41-1017-splice-bs.gguf'
a[a.index('--rank') + 1] = '/mnt/win/llm/splice-b/rank-combined.bin'
for k in ('STRATA_BLOB_TWIN', 'STRATA_BLOB_MIRROR_SHARE', 'STRATA_ROUTE_COUNTS'):
    p['env'].pop(k, None)
p['_comment'] = 'B-s splice (3.29 bpw) on the production layout of 2026-10-23: model + rank swapped, no blob twin, no route counts'
json.dump(p, open('local/ds41-yuya-3gpu-splice-bs.json', 'w'), indent=1, ensure_ascii=False)
PY
for row in "F-ref5 local/ds41-yuya-3gpu.json STRATA_ROUTE_COUNTS=" "S-bs local/ds41-yuya-3gpu-splice-bs.json" \
           "S-bs-arena local/ds41-yuya-3gpu-splice-bs.json STRATA_PEER_FILL=arena"; do
  set -- $row
  step $1 && continue
  echo "== sweep $1 start $(date '+%F %T')"
  local/sweep.sh "$@" || { echo "== sweep $1 FAILED"; break; }
  finished $1
done
(cd /home/yuya/ds41 && python3 table.py F-ref4 F-ref5 S-bs S-bs-arena)
echo local/ds41-yuya-3gpu.json > local/active-config
bash /home/yuya/dsh-work/tools/restart-ds41.sh
echo "== splice-chain2 end $(date '+%F %T') (service on local/ds41-yuya-3gpu.json)"
