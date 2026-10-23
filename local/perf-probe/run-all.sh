#!/bin/bash
# The performance probe: the production configuration against exact-output variants, on the engine's line protocol.
# Stops the server, runs the variants, STARTS THE SERVER AGAIN on the production config.  Report: $T/report.txt
#   V0 base      : the production config as is
#   V1 keep      : STRATA_SERVE_KEEP_CACHE=1 --no-vram-lru (a batch keeps the VRAM expert cache's head; no re-seed after it)
#   V2 passes    : STRATA_SERVE_PASS_BATCH=1 STRATA_SERVE_STEP_MAX=512 STRATA_SERVE_BATCH_MIN=0 (appends up to ~640 tokens as
#                  decode-side passes, the pass's 16 tokens' experts served once)
#   V3 lookup    : STRATA_SERVE_LOOKUP=8 (copies from the context as drafts, verified: the same output)
set -u
R=/home/yuya/llm-src/strata-ds41; T=${1:-/tmp/perf-probe}; PY=/home/yuya/llm-src/Strata-DSV4.1/.venv/bin/python
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
run() {   # name envs... [-- args]
  local name=$1; shift
  rm -rf "$T/$name"; mkdir -p "$T/$name"
  echo "$(date +%T) $name $*"
  "$PY" -I local/perf-probe/run.py "$T/$name" "$T/payload" "$name" "$@" > "$T/$name.out" 2>&1
  echo "$(date +%T) $name exit $?"
  wait_vram
}
echo "$(date +%T) payload"
"$PY" -I local/perf-probe/build_payload.py "$T/payload" > "$T/payload.out" 2>&1 || { echo "payload failed"; cat "$T/payload.out"; exit 1; }
echo "$(date +%T) stopping strata-ds41"
sudo systemctl stop strata-ds41.service
wait_vram
run V0_base
run V1_keep STRATA_SERVE_KEEP_CACHE=1 -- --no-vram-lru
run V2_passes STRATA_SERVE_PASS_BATCH=1 STRATA_SERVE_STEP_MAX=512 STRATA_SERVE_BATCH_MIN=0
run V3_lookup STRATA_SERVE_LOOKUP=8
{ for r in V0_base V1_keep V2_passes V3_lookup; do cat "$T/$r.out"; echo; done; } > "$T/report.txt" 2>&1
echo "$(date +%T) starting strata-ds41"
bash /home/yuya/dsh-work/tools/restart-ds41.sh
echo "$(date +%T) done"
