#!/bin/bash
# Post-chain tests of 2026-10-23: the LRU-aware keep-cache (build-hip2 engine) against production, and the shared-prefix
# files (a prefix saved by one engine start, hit by the next).  Stops the server, restarts it on the production config.
set -u
R=/home/yuya/llm-src/strata-ds41; T=${1:-/tmp/keep-test}; PY=/home/yuya/llm-src/Strata-DSV4.1/.venv/bin/python
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
echo "$(date +%T) payloads"
[ -f "$T/payload/sizes.json" ] || "$PY" -I local/perf-probe/build_payload.py "$T/payload" > "$T/payload.out" 2>&1 || exit 1
[ -f "$T/payload-slot/sizes.json" ] || "$PY" -I local/slot-test/build_payload.py "$T/payload-slot" > "$T/payload-slot.out" 2>&1 || exit 1
echo "$(date +%T) stopping strata-ds41"
sudo systemctl stop strata-ds41.service
wait_vram
probe() {   # name envs...
  local name=$1; shift
  rm -rf "$T/$name"; mkdir -p "$T/$name"
  echo "$(date +%T) probe $name $*"
  "$PY" -I local/perf-probe/run.py "$T/$name" "$T/payload" "$name" "$@" > "$T/$name.out" 2>&1
  echo "$(date +%T) $name exit $?"
  wait_vram
}
probe V0_base DS41_EXE=$NEW
probe V4_keep DS41_EXE=$NEW STRATA_SERVE_KEEP_CACHE=1
rm -rf "$T/pfx"; mkdir -p "$T/pfx"
for m in prefix prefix2; do
  rm -rf "$T/$m"; mkdir -p "$T/$m"
  echo "$(date +%T) slot-test $m"
  "$PY" -I local/slot-test/run.py "$T/$m" "$T/payload-slot" "$m" STRATA_SERVE_PARK_SLOTS=2 PROMPT_BATCH=8192 DS41_EXE=$NEW STRATA_SERVE_PREFIX_DIR=$T/pfx > "$T/$m.out" 2>&1
  echo "$(date +%T) $m exit $?"
  ls -la "$T/pfx"
  wait_vram
done
{ cat "$T/V0_base.out" "$T/V4_keep.out"; echo; cat "$T/prefix.out" "$T/prefix2.out"; echo
  "$PY" -I local/slot-test/compare.py "$T/prefix" - - "$T/prefix2"; } > "$T/report.txt" 2>&1
echo "$(date +%T) starting strata-ds41"
bash /home/yuya/dsh-work/tools/restart-ds41.sh
echo "$(date +%T) done"
