#!/bin/bash
# oneshot-1gpu.sh TAG [EXE] [ENV=VAL ...] - a single-GPU DSpark one-shot decode (40-token prompt, 64 tokens, small arena,
# no peers) whose verify-row logits go to logs/oneshot1-TAG/rows.bin (STRATA_DSPARK_DUMP) for a bit-exact A/B of changes
# on the finish / predict paths that the dense bench (tier off) does not exercise.  Service must be stopped.
cd /home/yuya/llm-src/strata-ds41
TAG=$1; shift
EXE=./build-hip/ds41_generate
if [ $# -gt 0 ] && [ "${1#*=}" = "$1" ]; then EXE=$1; shift; fi
O=logs/oneshot1-$TAG; mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm/lib
env HIP_VISIBLE_DEVICES=0 STRATA_NO_LARGEPAGES=1 STRATA_DSPARK_DUMP="$O/rows.bin" \
  STRATA_DSPARK=/home/yuya/models/dsv41-1017-strata/DSpark-1017-Q2K-Q4K-sidecar.gguf STRATA_DSPARK_DEPTH=1 "$@" \
  $EXE -m /mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf --ids-file local/eval-lucebox/prompt40.i32 -n 64 --temp 0 \
  --backend cuda --experts gpu --rank /home/yuya/models/dsv41-1017-strata/rank-lucebox.bin --threads 12 --ctx 8192 --kv-fp4 \
  --prompt-batch 2048 --arena-gib 6 --arena-adapt \
  --engram /home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
  --route-bias 0 --route-bias-file 0 > "$O/run.out" 2> "$O/run.log"
echo "== $TAG exit $?"; grep -E "^decode:|^DSpark:|^pass_v" "$O/run.log" | cut -c1-200; md5sum "$O/run.out" | cut -c1-12
