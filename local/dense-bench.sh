#!/bin/bash
# dense-bench.sh TAG R:P [EXE] [ENV=VAL ...] - the dense half alone (--dense-bench): single GPU, small arena, no peers,
# no DSpark.  Prints ms/pass, the per-call split and (STRATA_DENSE_PROF=1) build/alloc/compute/readback.  Logits of
# every bench row go to logs/dbench-TAG/rows.f32 for a bit-exact A/B (cmp).  Service must be stopped.
cd /home/yuya/llm-src/strata-ds41
TAG=$1; RP=$2; shift 2
EXE=./build-hip/ds41_generate
if [ $# -gt 0 ] && [ "${1#*=}" = "$1" ]; then EXE=$1; shift; fi
O=logs/dbench-$TAG; mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm/lib
env HIP_VISIBLE_DEVICES=0 STRATA_NO_LARGEPAGES=1 "$@" \
  $EXE -m /mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf --ids-file local/eval-lucebox/prompt40.i32 -n 1 --temp 0 \
  --backend cuda --experts gpu --rank /home/yuya/models/dsv41-1017-strata/rank-lucebox.bin --threads 12 --ctx 8192 --kv-fp4 \
  --prompt-batch 2048 --arena-gib 6 --arena-adapt \
  --engram /home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
  --route-bias 0 --route-bias-file 0 --dense-bench "$RP" --dump-last "$O/rows.f32" > "$O/run.out" 2> "$O/run.log"
echo "== $TAG ($RP) exit $?"; grep -E "^dense-bench|^dense prof|loaded in" "$O/run.log" | cut -c1-260
