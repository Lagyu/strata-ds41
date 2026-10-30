#!/bin/bash
# window2.sh - critical-path window 2: the matrix-core indexer (STRATA_INDEXER_WMMA) and the chunk-prefetch slab
# (STRATA_CHUNK_PF_MIN, two-drive reader) against window 1's baselines; the seed pinned-buffer fix; the Engram hot pack.
#   T1  test_ds41_indexer-wmma --large: vec vs wmma scores + timing (cmp_lidx.py)
#   W_long / PF_long / WPF_long   run-long (98K read + appends + bench) with the indexer kernel / the slab prefetch / both
#   PF_std                        the std probe with the slab prefetch (short context)
#   B2_std                        the std probe on the final binary (seed fix: cache-open), HOT_std = + the Engram hot pack
#   HO                            held-out 8x1024 with STRATA_INDEXER_WMMA=1 (ho-eval.sh; reference s027-FA2)
#   sudo systemd-run --unit=crit-window2 --collect --property=User=yuya --property=LimitMEMLOCK=infinity \
#     --property=LimitNOFILE=65536 /bin/bash /home/yuya/ds41/crit/window2.sh
set -u
R=/home/yuya/llm-src/strata-ds41; S=/home/yuya/ds41/crit; T=$S/arms2
PY=/home/yuya/llm-src/Strata-DSV4.1/.venv/bin/python
BIN=/home/yuya/llm-src/strata-zcnvis/build-hip-deploy/ds41_generate-c2
TEST=/home/yuya/llm-src/strata-zcnvis/build-hip-deploy/test_ds41_indexer-c2
CFG=$R/local/ds41-yuya-3gpu.json
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/yuya ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib
cd "$R" || exit 1
mkdir -p "$T"
exec >> "$S/window2.log" 2>&1
echo "== window2 start $(date '+%F %T') binary $(sha256sum "$BIN" | cut -c1-16)"
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
             'build-*' 'hw-ab*' 'vf*' 'zc-*' 'sec*' 'decprof*' 'crit-window1*' 2>/dev/null | awk '{print $1}' | grep -v -E '^(crit-window2)' ; }
for i in $(seq 1 480); do o=$(others); [ -z "$o" ] && break; [ $((i % 40)) = 1 ] && echo "$(date +%T) waiting for: $(echo $o)"; sleep 15; done
[ -z "$(others)" ] || { echo "other GPU units still active after 2 h - giving up"; exit 1; }
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
# ---- T1: the indexer kernels, values + timing
if [ ! -f "$T/T1.done" ]; then
  echo "$(date +%T) T1 indexer test"
  rm -f "$T/lidx-vec.bin" "$T/lidx-wmma.bin"
  LIDX_DUMP=$T/lidx-vec.bin LIDX_TIME=1 "$TEST" --cuda --device 0 --large > "$T/T1-vec.out" 2>&1; echo "vec exit $?"
  LIDX_DUMP=$T/lidx-wmma.bin LIDX_TIME=1 STRATA_INDEXER_WMMA=1 "$TEST" --cuda --device 0 --large > "$T/T1-wmma.out" 2>&1; echo "wmma exit $?"
  grep -E "PASS|FAIL|rerun|nk=262144|nk=131072" "$T/T1-vec.out" | tail -8; grep -E "PASS|FAIL|rerun|nk=262144|nk=131072|indexer:" "$T/T1-wmma.out" | tail -9
  python3 -I "$S/cmp_lidx.py" "$T/lidx-vec.bin" "$T/lidx-wmma.bin" | tee "$T/T1-cmp.txt"
  wait_free; touch "$T/T1.done"
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
# every arm on the c2 binary; F0 = the fused hc-mixes kernel off (window 1's dense path), so the slab arms compare to B2
F0=STRATA_HC_MIXES_FUSED=0
probe B2_long  $LONG $S/payload-long            $F0
probe PF5_long $LONG $S/payload-long            $F0 STRATA_CHUNK_PF_MIN=512 STRATA_CHUNK_PF_FRAC=0.5 -- --chunk-prefetch 4
probe PF_long  $LONG $S/payload-long            $F0 STRATA_CHUNK_PF_MIN=2048 -- --chunk-prefetch 8
probe B2_std   $STD  /home/yuya/ds41/crit/arms1/payload  $F0
probe PF5_std  $STD  /home/yuya/ds41/crit/arms1/payload  $F0 STRATA_CHUNK_PF_MIN=512 STRATA_CHUNK_PF_FRAC=0.5 -- --chunk-prefetch 4
probe C2_std   $STD  /home/yuya/ds41/crit/arms1/payload
probe C2_long  $LONG $S/payload-long
probe HOT_std  $STD  /home/yuya/ds41/crit/arms1/payload  $F0 STRATA_ENGRAM_HOT=/home/yuya/models/dsv41-1017-strata/engram-hot-4g.bin
probe CA_std   $STD  /home/yuya/ds41/crit/arms1/payload  $F0 STRATA_CHUNK_ASYNC=1
probe W_long2  $LONG $S/payload-long            $F0 STRATA_INDEXER_WMMA=1
# ---- HO: held-out quality of the matrix-core indexer (reference s027-FA2: FA + NVIS on the same eval settings)
if [ -f "$T/W_long2.done" ] && [ ! -f "$T/HO.done" ]; then   # HO only after the WMMA long read ran clean
  echo "$(date +%T) held-out eval crit-wmma"
  BIN=$BIN MODEL=/mnt/fast/llm/orca/DS41-1017-Mix256G.gguf TMO=1500 bash /home/yuya/ds41/s027/ho-eval.sh crit-wmma \
      STRATA_BLOB_TWIN=/home/yuya/twin/DS41-1017-Mix256G.gguf STRATA_BLOB_MIRROR_SHARE=0.40 STRATA_PRESTAGE_GIB=0 \
      STRATA_FA_WMMA512=1 STRATA_INDEXER_NVIS=1 STRATA_INDEXER_WMMA=1 -- --skip-file 0.10 > "$T/HO.out" 2>&1
  echo "$(date +%T) held-out exit $?: $(cat local/eval-lucebox/crit-wmma/score.json 2>/dev/null | tr -d '\n')"
  wait_free; touch "$T/HO.done"
fi
echo "$(date +%T) arms done"
