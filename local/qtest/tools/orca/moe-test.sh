#!/bin/bash
# moe-test.sh - after the build (BUILD OK) and the conversion (MERGE.done): every expert class of the Mix-256G file against a
# dequant + F32 reference on real experts, BESIDE production (small VRAM cache, 4 GiB arena per process). At most TWO processes
# at a time ... (see the comment at the stages).
#   layers 0 (IQ2_S/IQ3_XXS) 4 (IQ3_XXS/IQ2_S) 5 (IQ3_XXS/IQ3_XXS) 13 (IQ3_XXS/IQ3_S) 15 (IQ3_XXS/MXFP4) 23 (IQ3_S/MXFP4)
#   24 (MXFP4/MXFP4) 34 (MXFP4/IQ3_S)  = gate=up type / down type of every class in the file.
# One process at a time: parity on both cards, then HIP 1 and HIP 2 decode + 32-token chunk, then their prompt-MMQ path, then the CPU tier.
# Writes local/qtest/orca-moe-test.ok or .FAILED. Launch:
#   sudo systemd-run --unit=orca-moetest --property=User=yuya --property=LimitMEMLOCK=infinity --property=Nice=10 --collect \
#        /bin/bash /home/yuya/ds41/orca/moe-test.sh
set -u
S=/home/yuya/ds41/orca; D=/mnt/ext/models/orca-mix256; R=/home/yuya/llm-src/strata-ds41; B=$R/build-hip-mayas
M=/mnt/win/llm/orca/DS41-1017-Mix256G-00001-of-00003.gguf
OK=$R/local/qtest/orca-moe-test.ok; BAD=$R/local/qtest/orca-moe-test.FAILED
LAYERS=0,4,5,13,15,23,24,34
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/yuya
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec >> "$S/moe-test.log" 2>&1
LAN=/home/yuya/.config/strata-ds41/lan.env
[ -f "$LAN" ] && { set -a; . "$LAN"; set +a; }
echo "$(date '+%F %T') moe-test.sh start"
rm -f "${OK:?}" "${BAD:?}"
until [ -e "$D/MERGE.done" ]; do
  systemctl is-active -q orca-convert || { echo "conversion unit ended without MERGE.done" > "$BAD"; echo "conversion failed"; exit 1; }
  sleep 30
done
echo "$(date '+%F %T') converted file ready: $(cat "$D/MERGE.done")"
# Coordination with the other sessions' windows (e.g. /home/yuya/ds41/skipfile-ab.sh): wait until no experiment engine runs (the
# production service may) and, with the service down, the GPUs hold no memory, then keep local/serve-engine.log fresh while the GPU tests run (their idle gates read its mtime).
EL=$R/local/serve-engine.log
foreign_engines() {   # ds41_generate processes that are NOT the production service's engine (another session's experiment)
  local n=0 p
  for p in $(pgrep -x ds41_generate); do grep -q 'strata-ds41.service' /proc/$p/cgroup 2>/dev/null || n=$((n + 1)); done
  echo $n
}
svc_age() {   # seconds since the production service entered its current state (active: since its start; else: since it stopped)
  local st ts
  st=$(systemctl show strata-ds41.service -p ActiveState --value)
  if [ "$st" = active ]; then ts=$(systemctl show strata-ds41.service -p ActiveEnterTimestamp --value)
  else ts=$(systemctl show strata-ds41.service -p InactiveEnterTimestamp --value); fi
  [ -n "$ts" ] && echo $(( $(date +%s) - $(date -d "$ts" +%s) )) || echo 999999
}
gpus_free() {
  # The small tests may run BESIDE a loaded production (as on 10-24 for GSQ-RCO) but never while production is loading: its peer
  # stores take nearly all of the R9700's memory (a restore began 2 s after an earlier start of this script). So: production
  # active for >= 5 min, or production down for >= 20 min with empty GPUs; and never beside another session's experiment engine.
  local b age
  [ "$(foreign_engines)" = 0 ] || return 1
  age=$(svc_age)
  if systemctl is-active -q strata-ds41.service; then
    [ "$age" -ge 300 ] && curl -s -o /dev/null -m 5 ${STRATA_API_KEY:+-H "Authorization: Bearer $STRATA_API_KEY"} http://127.0.0.1:8150/v1/models
    return
  fi
  b=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk -F': ' '{if ($NF+0 > m) m = $NF+0} END{print m+0}')
  [ "$age" -ge 1200 ] && [ "${b:-100}" -le 3 ]
}
note=0
until gpus_free && sleep 20 && gpus_free && sleep 20 && gpus_free; do
  now=$(date +%s); [ $((now - note)) -ge 1800 ] && { echo "$(date +%T) waiting for free GPUs (another job is running)"; note=$now; }
  sleep 30
done
echo "$(date +%T) GPUs free"
WIN=$S/.window; touch "$WIN"
( while [ -e "$WIN" ]; do touch "$EL"; sleep 10; done ) &
GUARD=$!
trap 'rm -f "$WIN"; kill $GUARD 2>/dev/null' EXIT
rm -f "${S:?}"/moe-*.rc "${S:?}"/moe-*.out
one() { local lab=$1; shift; "$@" > "$S/moe-$lab.out" 2>&1; echo $? > "$S/moe-$lab.rc"; }
# ONE process at a time: the driver caps the pinned host memory of all processes together, and production holds 95 GiB of it
# (two parallel test processes died with "c_bounce: out of memory" on 10-24 and 10-25; one alone runs). No bare `wait` either:
# the log guard below is a background job that never ends.
echo "$(date +%T) A: kernel parity on both cards"
one parity-hip1 env HIP_VISIBLE_DEVICES=1 "$B/native_grouped_parity"
one parity-hip2 env HIP_VISIBLE_DEVICES=2 "$B/native_grouped_parity"
echo "$(date +%T) B: decode + 32-token chunk, HIP 1 then HIP 2"
one gpu-hip1 env HIP_VISIBLE_DEVICES=1 "$B/test_ds41_moe_gpu" "$M" --layers "$LAYERS" --tokens 6 --gpu --chunk 32
one gpu-hip2 env HIP_VISIBLE_DEVICES=2 "$B/test_ds41_moe_gpu" "$M" --layers "$LAYERS" --tokens 6 --gpu --chunk 32
echo "$(date +%T) B2: prompt MMQ path, HIP 1 then HIP 2"
one gpu-hip1-mmq env HIP_VISIBLE_DEVICES=1 "$B/test_ds41_moe_gpu" "$M" --layers "$LAYERS" --tokens 2 --gpu --chunk 32 --chunk-mmq
one gpu-hip2-mmq env HIP_VISIBLE_DEVICES=2 "$B/test_ds41_moe_gpu" "$M" --layers "$LAYERS" --tokens 2 --gpu --chunk 32 --chunk-mmq
echo "$(date +%T) C: CPU tier"
one cpu "$B/test_ds41_moe" "$M" --layers "$LAYERS" --tokens 6
fail=0
for lab in parity-hip1 gpu-hip1 gpu-hip1-mmq parity-hip2 gpu-hip2 gpu-hip2-mmq cpu; do
  rc=$(cat "$S/moe-$lab.rc" 2>/dev/null || echo missing)
  echo "== $lab exit $rc"
  grep -E "classes:|^tier |FAIL|^stats:|failures|out of memory|chunk [0-9]+ tokens" "$S/moe-$lab.out" | tail -12
  [ "$rc" = "0" ] || fail=1
done
if [ $fail -eq 0 ]; then
  { echo "Mix-256G expert test PASSED $(date '+%F %T') (layers $LAYERS: parity + HIP decode/chunk/MMQ on HIP 1 and 2 + CPU tier)"
    grep -h "^stats:" "$S"/moe-cpu.out "$S"/moe-gpu-hip*.out; } > "$OK"
  echo "$(date '+%F %T') ALL PASSED -> $OK"
else
  echo "see $S/moe-test.log" > "$BAD"; echo "$(date '+%F %T') FAILED -> $BAD"; exit 1
fi
