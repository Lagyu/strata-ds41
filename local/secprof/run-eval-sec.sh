#!/usr/bin/env bash
# run-eval-sec.sh TAG [MODEL] [-- EXTRA ENGINE ARGS] - the security held-out set (local/secprof/build.py: sec_eval.npy, 16 x 2048 tokens of
# held-out Juliet code, CWE entries and real code, run as 4 sequences of 8192) through the 3-card residency of
# run-eval-3gpu.sh (40 GiB arena, so it can share the box with a requantization), logits dumped.  TAG=prod writes the
# reference (eval/teacher-prod.npz) from the production file; any other TAG is scored against it (score_sec.py):
# KL(prod || TAG), top-1 agreement, both perplexities.  RANK=file overrides the placement order.
# The chat server must be stopped.
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=$1; shift
MODEL=/mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf
if [ $# -gt 0 ] && [ "$1" != "--" ]; then MODEL=$1; shift; fi
[ $# -gt 0 ] && [ "$1" = "--" ] && shift
EXTRA=("$@")   # [-- EXTRA ENGINE ARGS]: a rule to score, e.g. -- --skip-file 0.10
E=local/secprof
O=$E/eval/$TAG
B=/home/yuya/datasets/security-profile/build
PY=/home/yuya/llm-src/hfenv/bin/python
mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
$PY -c "
import numpy as np
t = np.load('$B/sec_eval.npy').reshape(4, -1)
np.save('$O/tokens.npy', t)
for i, s in enumerate(t): s.astype(np.int32).tofile('$O/seq%d.i32' % i)"
for i in 0 1 2 3; do
  env HIP_VISIBLE_DEVICES=0,1,2 STRATA_PF_NO_DUMMY=1 STRATA_NO_LARGEPAGES=1 STRATA_PEER_GIB=28,20 STRATA_PEER_FILL=file \
    ./build-hip/ds41_generate -m "$MODEL" --ids-file "$O/seq$i.i32" -n 1 --temp 0 \
    --backend cuda --experts gpu --arena-gib 40 --arena-adapt --peer-device 1,2 --ctx 16384 --kv-fp4 --threads 12 \
    --rank "${RANK:-/home/yuya/models/dsv41-1017-strata/rank-lucebox.bin}" \
    --engram /home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
    --route-bias 0 --route-bias-file 0 --ppl --dump-logits "$O/seq$i.f32" "${EXTRA[@]}" > "$O/seq$i.out" 2> "$O/seq$i.log"
  echo "seq $i: $(grep '^PPL' "$O/seq$i.out")"
done
if [ "$TAG" = prod ]; then
  $PY $E/score_sec.py teacher "$O" "$O/tokens.npy" $E/eval/teacher-prod.npz | tee "$O/score.json"
else
  $PY $E/score_sec.py score "$O" "$O/tokens.npy" $E/eval/teacher-prod.npz | tee "$O/score.json"
fi
rm -f "$O"/seq*.f32
