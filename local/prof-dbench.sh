#!/bin/bash
# prof-dbench.sh TAG R:P [ENV=VAL ...] - rocprofv3 kernel stats of a dense-only bench (local/dense-bench.sh workload)
cd /home/yuya/llm-src/strata-ds41
TAG=$1; RP=$2; shift 2
O=logs/pdbench-$TAG; rm -rf "$O"; mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm/lib
env HIP_VISIBLE_DEVICES=0 STRATA_NO_LARGEPAGES=1 "$@" \
  rocprofv3 --kernel-trace --stats -d "$O" -o prof -- \
  ./build-hip/ds41_generate -m /mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf --ids-file local/eval-lucebox/prompt40.i32 -n 1 --temp 0 \
  --backend cuda --experts gpu --rank /home/yuya/models/dsv41-1017-strata/rank-lucebox.bin --threads 12 --ctx 8192 --kv-fp4 \
  --prompt-batch 2048 --arena-gib 6 --arena-adapt \
  --engram /home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
  --route-bias 0 --route-bias-file 0 --dense-bench "$RP" > "$O/run.out" 2> "$O/run.log"
echo "== $TAG ($RP) exit $?"; grep -E "^dense-bench" "$O/run.log" | cut -c1-200
S=$(find "$O" -name "prof_kernel_stats.csv" | head -1)
python3 - "$S" <<'PY'
import csv, sys
rows=list(csv.DictReader(open(sys.argv[1])))
rows.sort(key=lambda r:-float(r['TotalDurationNs']))
print(f"{'calls':>7} {'total_ms':>9} {'avg_us':>8}  name")
for r in rows[:28]:
    print(f"{int(r['Calls']):7d} {float(r['TotalDurationNs'])/1e6:9.1f} {float(r['AverageNs'])/1e3:8.1f}  {r['Name'][:100]}")
PY
