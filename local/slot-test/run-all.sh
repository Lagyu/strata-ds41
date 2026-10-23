#!/bin/bash
# Validation of the shared-prefix parks, the END parks and the request slots on the engine's line protocol.  Stops the
# server, runs the three test engines in turn, then STARTS THE SERVER AGAIN: on the new binary when every check passed,
# on the saved production binary (build-hip/ds41_generate.prod-1022-2107) when not.  Report: $T/report.txt
set -u
R=/home/yuya/llm-src/strata-ds41; T=${1:-/tmp/slot-test}; PY=/home/yuya/llm-src/Strata-DSV4.1/.venv/bin/python
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
run() {   # name mode envs...
  local name=$1 mode=$2; shift 2
  rm -rf "$T/$name"; mkdir -p "$T/$name"
  echo "$(date +%T) $name ($mode) $*"
  "$PY" -I local/slot-test/run.py "$T/$name" "$T/payload" "$mode" "$@" > "$T/$name.out" 2>&1
  echo "$(date +%T) $name exit $?"
  wait_vram
}
echo "$(date +%T) payload"
"$PY" -I local/slot-test/build_payload.py "$T/payload" > "$T/payload.out" 2>&1 || { echo "payload failed"; cat "$T/payload.out"; exit 1; }
echo "$(date +%T) stopping strata-ds41"
sudo systemctl stop strata-ds41.service
wait_vram
B=PROMPT_BATCH=8192
run prefix prefix STRATA_SERVE_PARK_SLOTS=2 $B
run endpark endpark $B
run slots slots STRATA_SERVE_SLICE=3 $B
{ for r in prefix endpark slots; do echo "== $r"; cat "$T/$r.out"; echo; done
  "$PY" -I local/slot-test/compare.py "$T/prefix" "$T/endpark" "$T/slots"; } > "$T/report.txt" 2>&1
ok=$?
echo "$(date +%T) compare exit $ok"
if [ "$ok" != "0" ]; then
  echo "$(date +%T) checks failed: production goes back to the saved binary"
  cp -p build-hip/ds41_generate build-hip/ds41_generate.slots-failed
  cp -p build-hip/ds41_generate.prod-1022-2107 build-hip/ds41_generate.tmp && mv build-hip/ds41_generate.tmp build-hip/ds41_generate
fi
echo "$(date +%T) starting strata-ds41"
bash /home/yuya/dsh-work/tools/restart-ds41.sh
echo "$(date +%T) done (checks $ok)"
