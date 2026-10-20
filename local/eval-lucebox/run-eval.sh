#!/usr/bin/env bash
# run-eval.sh TAG [ENV=VAL ...] - Lucebox's 1017 held-out set (8 x 1024 tokens, heldout_8x1024.npy) through this
# engine on the MXFP4 1017 model: one run per sequence with --ppl --dump-logits, then score.py compares every row
# with the FP8 1017 teacher's top-256 log-probs (teacher_8x1024.npz; teacher ppl 8.56).  Exact routing; the extra
# ENV=VAL pairs go to the engine.  The chat server must be stopped (sudo systemctl stop strata-ds41).
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=$1; shift
E=local/eval-lucebox; O=$E/$TAG; mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python3 -c "
import numpy as np
t = np.load('$E/heldout_8x1024.npy')
for i, s in enumerate(t): s.astype(np.int32).tofile('$O/seq%d.i32' % i)"
for i in 0 1 2 3 4 5 6 7; do
  env HIP_VISIBLE_DEVICES=0 STRATA_PF_NO_DUMMY=1 STRATA_NO_LARGEPAGES=1 "$@" ./build-hip/ds41_generate \
    -m /mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf --ids-file "$O/seq$i.i32" -n 1 --temp 0 \
    --backend cuda --experts gpu --arena-gib 75 --ctx 4096 --kv-q8 --threads 12 \
    --rank /home/yuya/models/dsv41-1017-strata/rank-lucebox.bin \
    --engram /home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
    --route-bias 0 --route-bias-file 0 --ppl --dump-logits "$O/seq$i.f32" \
    > "$O/seq$i.out" 2> "$O/seq$i.log"
  echo "seq $i: $(grep '^PPL' "$O/seq$i.out")"
done
python3 $E/score.py "$O" | tee "$O/score.json"
rm -f "$O"/seq*.f32
