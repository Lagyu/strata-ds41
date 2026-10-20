#!/bin/bash
# oneshot-decode.sh TAG [ENV=VAL ...] - the prof-decode workload without the profiler: DSpark depth 1, 96 tokens after a 40-token
# prompt on the production layout; prints the decode rate and the DSpark verify / draft ms per pass.  Service must be stopped.
cd /home/yuya/llm-src/strata-ds41
TAG=$1; shift
EXE=./build-hip/ds41_generate
if [ $# -gt 0 ] && [ "${1#*=}" = "$1" ]; then EXE=$1; shift; fi
O=logs/oneshot-$TAG; mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm/lib
env HIP_VISIBLE_DEVICES=0,1,2 STRATA_NO_LARGEPAGES=1 STRATA_PEER_GIB=28,20 STRATA_PEER_FILL=file \
  STRATA_DSPARK=/home/yuya/models/dsv41-1017-strata/DSpark-1017-Q2K-Q4K-sidecar.gguf STRATA_DSPARK_DEPTH=1 "$@" \
  $EXE -m /mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf --ids-file ${IDS:-local/eval-lucebox/prompt40.i32} -n ${NGEN:-96} --temp 0 \
  --backend cuda --experts gpu --rank /home/yuya/models/dsv41-1017-strata/rank-lucebox.bin --threads 12 --ctx 8192 --kv-fp4 \
  --prompt-batch 2048 --arena-gib 95 --peer-device 1,2 --arena-adapt \
  --engram /home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
  --route-bias 0 --route-bias-file 0 > "$O/run.out" 2> "$O/run.log"
grep -E "^decode:|^DSpark|^pass_v|^prompt:" "$O/run.log" | cut -c1-200
