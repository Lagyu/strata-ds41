#!/usr/bin/env bash
# run-eval-mayas.sh TAG [ENV=VAL ...] [-- EXTRA ENGINE ARGS] - local/eval-lucebox/run-eval-3gpu.sh with the binary and the
# model as variables: BIN (default build-hip-mayas/ds41_generate) and MODEL (required). Same residency (XTX primary,
# R9700 28 GiB + XTX 20 GiB peer stores, 95 GiB arena, fp4 KV, exact routing), same 8 x 1024 held-out sequences and the
# same scorer, so the score is comparable with base-3gpu. The chat server must be stopped.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${BIN:-./build-hip-mayas/ds41_generate}
: "${MODEL:?MODEL=path to the merged Maya-S GGUF}"
TAG=$1; shift
ENVS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done
[ $# -gt 0 ] && shift
E=local/eval-lucebox; O=$E/$TAG; mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
/home/yuya/llm-src/hfenv/bin/python -c "
import numpy as np
t = np.load('$E/heldout_8x1024.npy')
for i, s in enumerate(t): s.astype(np.int32).tofile('$O/seq%d.i32' % i)"
for i in 0 1 2 3 4 5 6 7; do
  env HIP_VISIBLE_DEVICES=0,1,2 STRATA_PF_NO_DUMMY=1 STRATA_NO_LARGEPAGES=1 STRATA_PEER_GIB=28,20 STRATA_PEER_FILL=file \
    "${ENVS[@]}" timeout -k 20 "${TMO:-1500}" "$BIN" \
    -m "$MODEL" --ids-file "$O/seq$i.i32" -n 1 --temp 0 \
    --backend cuda --experts gpu --arena-gib 95 --arena-adapt --peer-device 1,2 --ctx 4096 --kv-fp4 --threads 12 \
    --rank /home/yuya/models/dsv41-1017-strata/rank-lucebox.bin \
    --engram /home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
    --route-bias 0 --route-bias-file 0 --ppl --dump-logits "$O/seq$i.f32" "$@" \
    > "$O/seq$i.out" 2> "$O/seq$i.log"
  echo "seq $i: $(grep '^PPL' "$O/seq$i.out") $(grep -o 'skipped [0-9]*' "$O/seq$i.log" | tail -1)"
done
/home/yuya/llm-src/hfenv/bin/python $E/score.py "$O" | tee "$O/score.json"
rm -f "$O"/seq*.f32
