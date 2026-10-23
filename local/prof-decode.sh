#!/bin/bash
# prof-decode.sh TAG [EXE] [ENV=VAL ...] - rocprofv3 kernel statistics + trace of a one-shot DSpark decode (depth 1, 96 tokens
# after a 40-token prompt) on the production layout (3 cards, peers, 95 GiB arena).  The service must be stopped.  Prints the
# engine's pass_v split (dense / tier per pass), the DSpark line, the top kernels and the dispatches per pass.
cd /home/yuya/llm-src/strata-ds41
TAG=$1; shift; EXE=./build-hip/ds41_generate; if [ $# -gt 0 ] && [ "${1#*=}" = "$1" ]; then EXE=$1; shift; fi   # EXE if given, then ENV=VAL
O=logs/prof-$TAG; rm -rf "$O"; mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm/lib
env HIP_VISIBLE_DEVICES=0,1,2 STRATA_NO_LARGEPAGES=1 STRATA_PEER_GIB=28,20 STRATA_PEER_FILL=file \
  STRATA_DSPARK=/home/yuya/models/dsv41-1017-strata/DSpark-1017-Q2K-Q4K-sidecar.gguf STRATA_DSPARK_DEPTH=1 "$@" \
  rocprofv3 --kernel-trace --memory-copy-trace --stats -d "$O" -o prof -f csv -- \
  "$EXE" -m /mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf --ids-file local/eval-lucebox/prompt40.i32 -n 96 --temp 0 \
  --backend cuda --experts gpu --rank /home/yuya/models/dsv41-1017-strata/rank-lucebox.bin --threads 12 --ctx 8192 --kv-fp4 \
  --prompt-batch 2048 --arena-gib 95 --peer-device 1,2 --arena-adapt \
  --engram /home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
  --route-bias 0 --route-bias-file 0 > "$O/run.out" 2> "$O/run.log"
grep -E "^pass_v split|^DSpark|^decode:|tok/s" "$O/run.log" | tail -4 | cut -c1-200
f=$(find "$O" -name "*kernel_stats.csv" | head -1); t=$(find "$O" -name "*kernel_trace.csv" | head -1)
python3 - "$f" "$t" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r["TotalDurationNs"]) for r in rows); calls = sum(int(r["Calls"]) for r in rows)
rows.sort(key=lambda r: -float(r["TotalDurationNs"]))
print(f"{'kernel':66s} {'calls':>7s} {'total ms':>9s} {'avg us':>7s} {'share':>6s}")
for r in rows[:28]:
    print(f"{r['Name'][:66]:66s} {int(r['Calls']):7d} {float(r['TotalDurationNs'])/1e6:9.1f} {float(r['TotalDurationNs'])/1e3/int(r['Calls']):7.1f} {100*float(r['TotalDurationNs'])/tot:5.1f}%")
print(f"total GPU kernel time {tot/1e9:.2f} s over {calls} dispatches")
# the trace: device busy vs wall on GPU 0 (agent of the first dispatch), by time
tr = list(csv.DictReader(open(sys.argv[2])))
if tr:
    k = [r for r in tr]
    agents = {}
    for r in k: agents.setdefault(r.get("Agent_Id", "?"), []).append((int(r["Start_Timestamp"]), int(r["End_Timestamp"])))
    for a, iv in agents.items():
        iv.sort(); busy = 0; cur_s, cur_e = iv[0]
        for s, e in iv[1:]:
            if s > cur_e: busy += cur_e - cur_s; cur_s, cur_e = s, e
            else: cur_e = max(cur_e, e)
        busy += cur_e - cur_s
        span = iv[-1][1] - iv[0][0]
        print(f"agent {a}: {len(iv)} dispatches, busy {busy/1e9:.2f} s of span {span/1e9:.2f} s ({100*busy/max(1,span):.0f}%)")
PY
