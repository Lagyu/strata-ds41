#!/bin/bash
# window1.sh - critical-path window 1 (2026-10-30 evening): the anatomy of prefill + decode with the profilers on, and the
# A/B of STRATA_TIER_ASYNC (the async layer sequence).  Production stops once quiet (no request in flight ~60 s), the EXIT
# trap starts it again.  Binary: the critpath build (production code + instrumentation + the async sequence, opt-in).
#   arms: B_std / A_std (perf-probe std payload: prose), B_sec / A_sec (code/security payload), B_long / A_long (98K read +
#         appends + 200-token bench), then ROCPROF: rocprofv3 kernel stats of the long read + one append (sync build).
#   sudo systemd-run --unit=crit-window1 --collect --property=User=yuya --property=LimitMEMLOCK=infinity \
#     --property=LimitNOFILE=65536 /bin/bash /home/yuya/ds41/crit/window1.sh
set -u
R=/home/yuya/llm-src/strata-ds41; S=/home/yuya/ds41/crit; T=$S/arms1
PY=/home/yuya/llm-src/Strata-DSV4.1/.venv/bin/python
BIN=/home/yuya/llm-src/strata-zcnvis/build-hip-deploy/ds41_generate
CFG=$R/local/ds41-yuya-3gpu.json
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/yuya ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib
cd "$R" || exit 1
mkdir -p "$T"
exec >> "$S/window1.log" 2>&1
echo "== window1 start $(date '+%F %T') binary $(sha256sum "$BIN" | cut -c1-16)"
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
             'build-*' 'hw-ab*' 'vf*' 'zc-*' 'sec*' 'decprof*' 2>/dev/null | awk '{print $1}' | grep -v -E '^(crit-window1)' ; }
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
rocm-smi --showpids 2>/dev/null | grep -q 'No KFD PIDs' || { echo "another process holds a GPU"; exit 1; }
[ -s "$T/payload/sizes.json" ] || "$PY" -I local/perf-probe/build_payload.py "$T/payload" > "$T/payload.out" 2>&1 || { echo "payload failed"; exit 1; }
PROF="STRATA_PASS_PROF=1 STRATA_DENSE_PROF=1 MIMO_CHUNK_PROF=1 STRATA_SEED_PROF=1"
probe() {   # name runner payload [KEY=VALUE ...]
  local name=$1 runner=$2 pl=$3; shift 3
  [ -f "$T/$name.done" ] && return 0
  rm -rf "$T/$name" "$T/pfx-$name"; mkdir -p "$T/$name"
  echo "$(date +%T) probe $name payload=$(basename $pl) $*"
  PROBE_CONFIG=$CFG timeout -k 30 1800 "$PY" -I "$runner" "$T/$name" "$pl" "$name" DS41_EXE="$BIN" \
      STRATA_SERVE_PREFIX_DIR="$T/pfx-$name" STRATA_TEXT_LOG=0 $PROF "$@" > "$T/$name.out" 2>&1
  local rc=$?
  [ $rc = 124 ] || [ $rc = 137 ] && { echo "$(date +%T) probe $name TIMED OUT - killing the engine"; pkill -x ds41_generate; sleep 5; }
  echo "$(date +%T) probe $name exit $rc"
  grep -aE '^\[[A-Za-z0-9_]+\] (A|bench|copy|warm|long)' "$T/$name.out" | cut -c1-200
  grep -aE 'serve: pass prof' "$T/$name/engine.log" | tail -3 | cut -c1-260
  wait_free
  [ $rc = 0 ] && touch "$T/$name.done"
  return $rc
}
STD=local/perf-probe/run.py; LONG=$S/run-long.py
probe B_std  $STD  "$T/payload"
probe A_std  $STD  "$T/payload"                 STRATA_TIER_ASYNC=1
probe B_sec  $STD  /home/yuya/ds41/decprof/payload-sec
probe A_sec  $STD  /home/yuya/ds41/decprof/payload-sec  STRATA_TIER_ASYNC=1
probe B_long $LONG $S/payload-long
probe A_long $LONG $S/payload-long              STRATA_TIER_ASYNC=1
# rocprofv3 kernel statistics of the long read + one append (dense-phase attribution at ~98K context): the engine itself
# runs under rocprofv3 (DS41_EXE = the wrapper), the runner outside it
if [ ! -f "$T/ROCPROF.done" ]; then
  echo "$(date +%T) rocprof long read"
  rm -rf "$T/ROCPROF" "$T/pfx-ROCPROF"; mkdir -p "$T/ROCPROF/trace"
  PROBE_CONFIG=$CFG ROCPROF_DIR="$T/ROCPROF/trace" timeout -k 30 2400 "$PY" -I "$S/run-long-short.py" "$T/ROCPROF" "$S/payload-long" ROCPROF \
      DS41_EXE="$S/rocprof-engine.sh" STRATA_SERVE_PREFIX_DIR="$T/pfx-ROCPROF" STRATA_TEXT_LOG=0 > "$T/ROCPROF.out" 2>&1
  echo "$(date +%T) rocprof exit $?"
  grep -aE '^\[ROCPROF\]' "$T/ROCPROF.out" | cut -c1-200
  wait_free
  touch "$T/ROCPROF.done"
fi
echo "$(date +%T) arms done"
