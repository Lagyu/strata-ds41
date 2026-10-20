#!/bin/bash
# prof-prefill.sh TAG [EXE] - rocprofv3 kernel statistics of a one-shot 1024-token prefill on GPU 0 (the service must be
# stopped): which attention kernel runs and how the dense phase splits by kernel.  Output: logs/prof-TAG/ (kernel_stats.csv).
cd /home/yuya/llm-src/strata-ds41
TAG=$1; EXE=${2:-./build-hip/ds41_generate}
O=logs/prof-$TAG; rm -rf "$O"; mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm/lib
env HIP_VISIBLE_DEVICES=0 STRATA_PF_NO_DUMMY=1 STRATA_NO_LARGEPAGES=1 \
  rocprofv3 --kernel-trace --stats -d "$O" -o prof --output-format csv -- \
  "$EXE" -m /mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf --ids-file local/eval-lucebox/base-3gpu/seq0.i32 -n 1 --temp 0 \
  --backend cuda --experts gpu --arena-gib 75 --ctx 4096 --kv-fp4 --threads 12 --prompt-batch 2048 \
  --rank /home/yuya/models/dsv41-1017-strata/rank-lucebox.bin \
  --engram /home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
  --route-bias 0 --route-bias-file 0 > "$O/run.out" 2> "$O/run.log"
grep -E "^prompt: |^prompt split" "$O/run.log" | cut -c1-160
f=$(find "$O" -name "*kernel_stats.csv" | head -1)
echo "kernel stats: $f"
python3 - "$f" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r["TotalDurationNs"]) for r in rows)
rows.sort(key=lambda r: -float(r["TotalDurationNs"]))
print(f"{'kernel':70s} {'calls':>7s} {'total ms':>10s} {'share':>6s}")
for r in rows[:22]:
    print(f"{r['Name'][:70]:70s} {int(r['Calls']):7d} {float(r['TotalDurationNs'])/1e6:10.1f} {100*float(r['TotalDurationNs'])/tot:5.1f}%")
print(f"total GPU kernel time {tot/1e9:.2f} s")
PY
