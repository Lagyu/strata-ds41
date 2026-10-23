#!/bin/bash
# run-final.sh [OUT_DIR] [extra probe envs...] - the cancel test (STOP during a long read) and the DSpark confidence cut
# (STRATA_DSPARK_CONF=0.5, Maya v1.1.20's cumulative rule at draft depth 1) on build-hip2's engine; the probe's extra envs
# (e.g. STRATA_SERVE_KEEP_CACHE=1) apply to both probe variants so the cut is measured against its own baseline.
# Stops the server, starts it again on the production config.  Report: $T/report.txt
set -u
R=/home/yuya/llm-src/strata-ds41; T=${1:-/tmp/final-test}; shift || true
PY=/home/yuya/llm-src/Strata-DSV4.1/.venv/bin/python
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/yuya
cd "$R" || exit 1
mkdir -p "$T"
exec > "$T/run-all.log" 2>&1
wait_vram() {
  for i in $(seq 1 90); do
    busy=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk -F': ' '{s+=$NF} END{print s+0}')
    [ "$busy" = "0" ] && break
    sleep 2
  done
  echo "$(date +%T) VRAM released after ~$((i * 2)) s"
}
NEW=$R/build-hip2/ds41_generate
[ -x "$NEW" ] || { echo "no $NEW"; exit 1; }
[ -f "$T/payload/sizes.json" ] || "$PY" -I local/perf-probe/build_payload.py "$T/payload" > "$T/payload.out" 2>&1 || exit 1
[ -f "$T/payload-slot/sizes.json" ] || "$PY" -I local/slot-test/build_payload.py "$T/payload-slot" > "$T/payload-slot.out" 2>&1 || exit 1
echo "$(date +%T) stopping strata-ds41"
sudo systemctl stop strata-ds41.service
wait_vram
rm -rf "$T/cancel"; mkdir -p "$T/cancel"
echo "$(date +%T) slot-test cancel"
CANCEL_AFTER_S=9 "$PY" -I local/slot-test/run.py "$T/cancel" "$T/payload-slot" cancel STRATA_SERVE_PARK_SLOTS=2 PROMPT_BATCH=8192 DS41_EXE=$NEW "$@" > "$T/cancel.out" 2>&1
echo "$(date +%T) cancel exit $?"
wait_vram
probe() {   # name envs...
  local name=$1; shift
  rm -rf "$T/$name"; mkdir -p "$T/$name"
  echo "$(date +%T) probe $name $*"
  "$PY" -I local/perf-probe/run.py "$T/$name" "$T/payload" "$name" "$@" > "$T/$name.out" 2>&1
  echo "$(date +%T) $name exit $?"
  wait_vram
}
probe V7_base DS41_EXE=$NEW "$@"
probe V6_conf DS41_EXE=$NEW STRATA_DSPARK_CONF=0.5 "$@"
{ cat "$T/cancel.out"; echo; "$PY" -I local/slot-test/compare.py - - - - "$T/cancel"; echo; cat "$T/V7_base.out" "$T/V6_conf.out"; } > "$T/report.txt" 2>&1
echo "$(date +%T) starting strata-ds41"
bash /home/yuya/dsh-work/tools/restart-ds41.sh
echo "$(date +%T) done"
