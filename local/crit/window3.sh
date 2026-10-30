#!/bin/bash
# window3.sh - critical-path window 3: the routing-hinted file-tier read stage (STRATA_CHUNK_HINT_GIB, binary c3) alone and
# the full candidate (matrix-core indexer + fused hc kernels + skip-read + hint) at short / security / 98K context, with
# same-window baselines; the held-out eval of the full candidate at the end (the deploy gate).
#   SM_off / SM_hint   one-shot 16K prefill + 24 greedy tokens without / with the hint stage: the outputs must be identical
#   H4_std B3_std      std probe: hint 4 GiB vs nothing (fused hc kernels off = window 2's dense path); SK0_std = skip-read off; D2_std = two copy streams; SF5_std = half seed (free slots filled by decode misses)
#   ALL_std ALL_sec    the full candidate (indexer + hint + fused hc + skip-read + chunk async) on the std / security payloads
#   H4_long B3_long ALL_long   run-long (98K read + appends + bench)
#   HO_all             held-out 8x1024 with the full candidate (reference zcnvis-deploy 0.10137 / 87.90%)
#   sudo systemd-run --unit=crit-window3 --collect --property=User=yuya --property=LimitMEMLOCK=infinity \
#     --property=LimitNOFILE=65536 /bin/bash /home/yuya/ds41/crit/window3.sh
set -u
R=/home/yuya/llm-src/strata-ds41; S=/home/yuya/ds41/crit; T=$S/arms3
PY=/home/yuya/llm-src/Strata-DSV4.1/.venv/bin/python
BIN=/home/yuya/llm-src/strata-zcnvis/build-hip-deploy/ds41_generate-c3
TEST=/home/yuya/llm-src/strata-zcnvis/build-hip-deploy/test_ds41_indexer-c2
CFG=$R/local/ds41-yuya-3gpu.json
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/yuya ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib
cd "$R" || exit 1
mkdir -p "$T"
exec >> "$S/window3.log" 2>&1
echo "== window3 start $(date '+%F %T') binary $(sha256sum "$BIN" | cut -c1-16)"
wait_free() {
  local b=x
  for i in $(seq 1 90); do
    b=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk -F': ' '{s+=$NF} END{print s+0}')
    [ "$b" = "0" ] && return 0
    sleep 2
  done
  echo "$(date +%T) WARNING: VRAM not released (sum of VRAM% $b)"
}
STOPPED=0
finish() {
  [ "$STOPPED" = 1 ] || { echo "== window end $(date '+%F %T') (production untouched)"; return; }
  if pgrep -f 'ds41_generate[-a-z0-9]* --serve' > /dev/null; then echo "$(date +%T) a window engine is still alive - killing it"; pkill -f 'ds41_generate[-a-z0-9]* --serve'; sleep 10; fi
  wait_free
  sudo -n systemctl start strata-ds41.service
  local k code=0 i
  k=$(grep -E '^STRATA_API_KEY=' /home/yuya/.config/strata-ds41/lan.env 2>/dev/null | cut -d= -f2-)
  for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $k" http://127.0.0.1:8150/v1/models 2>/dev/null)
    [ "$code" = 200 ] && break
    systemctl is-active -q strata-ds41.service || break
    sleep 5
  done
  echo "$(date +%T) production start: API $code after ~$((i * 5)) s ($(systemctl is-active strata-ds41.service))"
  echo "== window end $(date '+%F %T')"
}
trap 'exit 143' TERM INT HUP
trap finish EXIT
others() { systemctl list-units --no-legend --plain --state=active,activating '*window*' '*post*' '*deploy*' '*verify*' 's027-*' \
             'build-*' 'hw-ab*' 'vf*' 'zc-*' 'sec*' 'decprof*' 'crit-window1*' 2>/dev/null | awk '{print $1}' | grep -v -E '^(crit-window3)' ; }
for i in $(seq 1 960); do o=$(others); [ -z "$o" ] && break; [ $((i % 40)) = 1 ] && echo "$(date +%T) waiting for: $(echo $o)"; sleep 15; done
[ -z "$(others)" ] || { echo "other GPU units still active after 4 h - giving up"; exit 1; }
quiet() {
  local age=$(( $(date +%s) - $(stat -c %Y local/serve-engine.log 2>/dev/null || echo 0) ))
  [ "$age" -ge 60 ] || return 1
  rocm-smi --showuse 2>/dev/null | awk -F': ' '/GPU use/ { if ($NF + 0 > 10) b = 1 } END { exit b }'
}
if systemctl is-active --quiet strata-ds41.service; then
  for i in $(seq 1 2880); do quiet && break; sleep 15; done
  [ -z "$(others)" ] || { echo "another GPU unit started meanwhile - not stopping production"; exit 1; }
  echo "$(date +%T) server quiet -> stopping production"
  sudo -n systemctl stop strata-ds41.service
fi
STOPPED=1
wait_free
# the stopped engine lingers in the KFD process list for a while after its VRAM is released (the pinned arena's teardown)
for i in $(seq 1 90); do rocm-smi --showpids 2>/dev/null | grep -q 'No KFD PIDs' && break; sleep 2; done
rocm-smi --showpids 2>/dev/null | grep -q 'No KFD PIDs' || { echo "another process holds a GPU: $(rocm-smi --showpids 2>/dev/null | grep -E '^[0-9]+' | head -2)"; exit 1; }
# ---- DMA: pinned host -> device bandwidth on 1 / 2 / 4 streams (the chunk path's and the decode tier's copy ceiling)
if [ ! -f "$T/DMA.done" ]; then
  echo "$(date +%T) dmabench (R9700)"
  HIP_VISIBLE_DEVICES=0 timeout 120 /home/yuya/ds41/crit/dmabench/dmabench 15 8 2>&1 | tee "$T/DMA.out"
  HIP_VISIBLE_DEVICES=0 timeout 120 /home/yuya/ds41/crit/dmabench/dmabench 1 2 2>&1 | tee -a "$T/DMA.out"
  echo "-- HSA_ENABLE_SDMA=0 (blit kernels instead of the SDMA engines)" | tee -a "$T/DMA.out"
  HSA_ENABLE_SDMA=0 HIP_VISIBLE_DEVICES=0 timeout 120 /home/yuya/ds41/crit/dmabench/dmabench 15 8 2>&1 | tee -a "$T/DMA.out"
  wait_free; touch "$T/DMA.done"
fi
PROF="STRATA_PASS_PROF=1 STRATA_DENSE_PROF=1 MIMO_CHUNK_PROF=1 STRATA_SEED_PROF=1"
probe() {   # name runner payload [KEY=VALUE ...]
  local name=$1 runner=$2 pl=$3; shift 3
  [ -f "$T/$name.done" ] && return 0
  rm -rf "$T/$name" "$T/pfx-$name"; mkdir -p "$T/$name"
  echo "$(date +%T) probe $name payload=$(basename $pl) $*"
  PROBE_CONFIG=$CFG timeout -k 30 1800 "$PY" -I "$runner" "$T/$name" "$pl" "$name" DS41_EXE="$BIN" \
      STRATA_SERVE_PREFIX_DIR="$T/pfx-$name" STRATA_TEXT_LOG=0 $PROF "$@" > "$T/$name.out" 2>&1   # "$@" may end with -- engine args
  local rc=$?
  [ $rc = 124 ] || [ $rc = 137 ] && { echo "$(date +%T) probe $name TIMED OUT - killing the engine"; pkill -f 'ds41_generate[-a-z0-9]* --serve'; sleep 5; }
  echo "$(date +%T) probe $name exit $rc"
  if pgrep -f 'ds41_generate[-a-z0-9]* --serve' > /dev/null; then echo "$(date +%T) the probe's engine outlived it - killing it"; pkill -f 'ds41_generate[-a-z0-9]* --serve'; sleep 10; fi
  grep -aE '^\[[A-Za-z0-9_]+\] (A|bench|copy|warm|long)' "$T/$name.out" | cut -c1-200
  grep -aE 'serve: pass prof' "$T/$name/engine.log" | tail -2 | cut -c1-200
  grep -aE '^chunk prof:|^chunk prefetch:' "$T/$name/engine.log" | tail -3 | cut -c1-250
  wait_free
  [ $rc = 0 ] && touch "$T/$name.done"
  return $rc
}
STD=local/perf-probe/run.py; LONG=$S/run-long.py
SEC=/home/yuya/ds41/decprof/payload-sec; PL=/home/yuya/ds41/crit/arms1/payload
F0=STRATA_HC_MIXES_FUSED=0
HINT=STRATA_CHUNK_HINT_GIB=4
ALL="STRATA_INDEXER_WMMA=1 STRATA_CHUNK_HINT_GIB=4 STRATA_CHUNK_ASYNC=1"
# ---- SM: the one-shot smoke + exactness check of the hint stage (16K Gutenberg prefill, 24 greedy tokens)
smoke() {   # name [KEY=VALUE ...]
  local name=$1; shift
  [ -f "$T/$name.done" ] && return 0
  . "$S/oneshot-env.sh"
  echo "$(date +%T) smoke $name $*"
  eval "env $ENV MIMO_CHUNK_PROF=1 STRATA_DENSE_PROF=1 $* timeout -k 20 900 $BIN $ARGS --ids-file $S/l16k.i32 -n 24 --temp 0" > "$T/$name.out" 2> "$T/$name.log"
  local rc=$?
  echo "$(date +%T) smoke $name exit $rc, $(wc -c < "$T/$name.out") bytes of output"
  grep -aE '^chunk hint:|^chunk prof:|^chunk prefetch:|hint: off' "$T/$name.log" | tail -4 | cut -c1-250
  wait_free
  [ $rc = 0 ] && touch "$T/$name.done"
}
smoke SM_off  STRATA_INDEXER_WMMA=1
smoke SM_hint STRATA_INDEXER_WMMA=1 $HINT
if cmp -s "$T/SM_off.out" "$T/SM_hint.out"; then echo "hint exactness: the 24-token outputs are IDENTICAL"; else echo "hint exactness: the outputs DIFFER"; fi
probe H4_std   $STD  $PL         $F0 $HINT
probe B3_std   $STD  $PL         $F0
probe SK0_std  $STD  $PL         $F0 STRATA_CHUNK_SKIP_READ=0
probe D2_std   $STD  $PL         $F0 STRATA_CHUNK_DMA_STREAMS=2
probe SF5_std  $STD  $PL         $F0 STRATA_SEED_FRAC=0.5
probe ALL_std  $STD  $PL         $ALL
probe ALL_sec  $STD  $SEC        $ALL
probe H4_long  $LONG $S/payload-long  $F0 $HINT
probe B3_long  $LONG $S/payload-long  $F0
probe ALL_long $LONG $S/payload-long  $ALL
# ---- HO: held-out quality of the full candidate (the deploy gate)
if [ -f "$T/ALL_long.done" ] && [ ! -f "$T/HO_all.done" ]; then
  echo "$(date +%T) held-out eval crit-all"
  BIN=$BIN MODEL=/mnt/fast/llm/orca/DS41-1017-Mix256G.gguf TMO=1500 bash /home/yuya/ds41/s027/ho-eval.sh crit-all \
      STRATA_BLOB_TWIN=/home/yuya/twin/DS41-1017-Mix256G.gguf STRATA_BLOB_MIRROR_SHARE=0.40 STRATA_PRESTAGE_GIB=0 \
      STRATA_FA_WMMA512=1 STRATA_INDEXER_NVIS=1 $ALL -- --skip-file 0.10 > "$T/HO_all.out" 2>&1
  echo "$(date +%T) held-out exit $?: $(cat local/eval-lucebox/crit-all/score.json 2>/dev/null | tr -d '\n')"
  wait_free; touch "$T/HO_all.done"
fi
echo "$(date +%T) arms done"
