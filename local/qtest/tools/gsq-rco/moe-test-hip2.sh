#!/bin/bash
# moe-test-hip2.sh - the XTX (HIP 2) half of moe-test.sh again, ALONE: in the parallel run its pinned bounce buffer failed
# ("ds4_moe: c_bounce: out of memory", exit 134) because the CPU test and the HIP 1 test held their arenas at the same time
# beside production's ~109 GB of locked memory. CPU, HIP 1 and both parity runs had passed. On success this writes the OK
# marker for the chain (and removes the FAILED one). Launch:
#   sudo systemd-run --unit=gsq-moetest2 --property=User=yuya --property=LimitMEMLOCK=infinity --property=Nice=10 --collect \
#        /bin/bash /home/yuya/ds41/gsq-rco/moe-test-hip2.sh
set -u
S=/home/yuya/ds41/gsq-rco; R=/home/yuya/llm-src/strata-ds41; B=$R/build-hip-mayas
M=/mnt/win/llm/qtest/DS41-1017-GSQRCO-IQ2XS-ABL.gguf
OK=$R/local/qtest/gsq-moe-test.ok; BAD=$R/local/qtest/gsq-moe-test.FAILED
LAYERS=0,8,9,11,17,23,24,25,30,31
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/yuya
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec >> "$S/moe-test.log" 2>&1
echo "$(date '+%F %T') moe-test-hip2.sh: HIP 2 alone"
fail=0
for lab in gpu-hip2 gpu-hip2-mmq; do
  if [ $lab = gpu-hip2 ]; then a=(--tokens 8 --gpu --chunk 32); else a=(--tokens 2 --gpu --chunk 32 --chunk-mmq); fi
  HIP_VISIBLE_DEVICES=2 "$B/test_ds41_moe_gpu" "$M" --layers "$LAYERS" "${a[@]}" > "$S/moe-$lab.out" 2>&1
  rc=$?; echo $rc > "$S/moe-$lab.rc"
  echo "== $lab exit $rc"; grep -E "^tier |FAIL|^stats:|chunk [0-9]+ tokens|out of memory" "$S/moe-$lab.out" | tail -13
  [ $rc -eq 0 ] || fail=1
done
for lab in cpu parity-hip1 gpu-hip1 gpu-hip1-mmq parity-hip2; do [ "$(cat "$S/moe-$lab.rc" 2>/dev/null)" = "0" ] || fail=1; done
if [ $fail -eq 0 ]; then
  { echo "GSQ-RCO IQ2_XS expert test PASSED $(date '+%F %T') (layers $LAYERS; parity + CPU + HIP decode/chunk/MMQ on HIP 1 and 2; HIP 2 rerun alone)"
    grep -h "^stats:" "$S"/moe-cpu.out "$S"/moe-gpu-hip*.out; } > "$OK"
  rm -f "${BAD:?}"
  echo "$(date '+%F %T') ALL PASSED -> $OK"
else
  echo "HIP 2 rerun failed: see $S/moe-test.log" > "$BAD"; echo "$(date '+%F %T') FAILED"; exit 1
fi
