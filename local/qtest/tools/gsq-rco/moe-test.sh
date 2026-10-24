#!/bin/bash
# moe-test.sh - after the build (BUILD OK) and the merge (MERGE.done): validates the new IQ2_XXS / IQ2_XS down kernels and
# every expert class of the GSQ-RCO IQ2_XS file BESIDE production (peer cards only, small VRAM cache, 4 GiB arena each):
#   native_grouped_parity on HIP 1 (R9700 gfx1201) and HIP 2 (XTX gfx1100);
#   test_ds41_moe (CPU tier) and test_ds41_moe_gpu (HIP tier: decode path + a 32-token prompt chunk, then the prompt MMQ path)
#   on the merged file against a dequant + F32 reference, one layer of each of the 10 (gate=up, down) classes; the two cards
#   run in parallel (the test's F32 reference re-dequantizes every expert per token: a 256-token chunk took far too long).
# Writes local/qtest/gsq-moe-test.ok (all pass) or local/qtest/gsq-moe-test.FAILED. Launch:
#   sudo systemd-run --unit=gsq-moetest --property=User=yuya --property=LimitMEMLOCK=infinity --property=Nice=10 \
#        --collect /bin/bash /home/yuya/ds41/gsq-rco/moe-test.sh
set -u
S=/home/yuya/ds41/gsq-rco; D=/mnt/ext/models/gsq-rco-1017; R=/home/yuya/llm-src/strata-ds41; B=$R/build-hip-mayas
M=/mnt/win/llm/qtest/DS41-1017-GSQRCO-IQ2XS-ABL.gguf
OK=$R/local/qtest/gsq-moe-test.ok; BAD=$R/local/qtest/gsq-moe-test.FAILED
LAYERS=0,8,9,11,17,23,24,25,30,31   # (16,16) (16,22) (17,17) (16,17) (22,17) (17,18) (17,22) (22,22) (22,16) (17,16)
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/yuya
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
L=$S/moe-test.log
exec >> "$L" 2>&1
echo "$(date '+%F %T') moe-test.sh start"
until grep -q "BUILD OK" "$S/build.log" 2>/dev/null; do
  grep -q "BUILD FAILED" "$S/build.log" 2>/dev/null && { echo "build failed" > "$BAD"; echo "build failed"; exit 1; }
  sleep 30
done
until [ -e "$D/MERGE.done" ]; do
  systemctl is-active -q gsq-merge || { echo "merge unit ended without MERGE.done" > "$BAD"; echo "merge failed"; exit 1; }
  sleep 30
done
echo "$(date '+%F %T') merged file ready: $(cat "$D/MERGE.done")"
one() {   # label cmd... -> $S/moe-<label>.out, exit code in $S/moe-<label>.rc
  local lab=$1; shift
  "$@" > "$S/moe-$lab.out" 2>&1
  echo $? > "$S/moe-$lab.rc"
}
seq_dev() {   # dev: the card's three runs in order
  local dev=$1
  one parity-hip$dev env HIP_VISIBLE_DEVICES=$dev "$B/native_grouped_parity"
  one gpu-hip$dev env HIP_VISIBLE_DEVICES=$dev "$B/test_ds41_moe_gpu" "$M" --layers "$LAYERS" --tokens 8 --gpu --chunk 32
  one gpu-hip$dev-mmq env HIP_VISIBLE_DEVICES=$dev "$B/test_ds41_moe_gpu" "$M" --layers "$LAYERS" --tokens 2 --gpu --chunk 32 --chunk-mmq
}
rm -f "${S:?}"/moe-*.rc
echo "$(date +%T) CPU tier + HIP 1 + HIP 2 in parallel"
one cpu "$B/test_ds41_moe" "$M" --layers "$LAYERS" --tokens 8 &
seq_dev 1 &
seq_dev 2 &
wait
fail=0
for lab in cpu parity-hip1 gpu-hip1 gpu-hip1-mmq parity-hip2 gpu-hip2 gpu-hip2-mmq; do
  rc=$(cat "$S/moe-$lab.rc" 2>/dev/null || echo missing)
  echo "== $lab exit $rc"
  grep -E "classes:|^tier |FAIL|^stats:|failures|chunk [0-9]+ tokens" "$S/moe-$lab.out" | tail -14
  [ "$rc" = "0" ] || fail=1
done
if [ $fail -eq 0 ]; then
  { echo "GSQ-RCO IQ2_XS expert test PASSED $(date '+%F %T') (layers $LAYERS; parity + CPU + HIP decode/chunk/MMQ on HIP 1 and 2)"
    grep -h "^stats:" "$S"/moe-cpu.out "$S"/moe-gpu-hip*.out; } > "$OK"
  echo "$(date '+%F %T') ALL PASSED -> $OK"
else
  echo "see $L" > "$BAD"; echo "$(date '+%F %T') FAILED -> $BAD"; exit 1
fi
