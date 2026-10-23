#!/usr/bin/env bash
# run-eval-sec-mayas.sh TAG MODEL - local/secprof/run-eval-sec.sh with the binary as a variable (BIN, default
# build-hip-mayas/ds41_generate): the security held-out set (16 x 2048 tokens of held-out Juliet code, CWE entries and real
# code, run as 4 sequences of 8192) through the 3-card residency, scored against production's reference
# local/secprof/eval/teacher-prod.npz (made by `run-eval-sec.sh prod`): KL(prod || TAG), top-1 agreement, both ppl.
# The chat server must be stopped.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${BIN:-./build-hip-mayas/ds41_generate}
TAG=$1
MODEL=$2
E=local/secprof
O=$E/eval/$TAG
B=/home/yuya/datasets/security-profile/build
PY=/home/yuya/llm-src/hfenv/bin/python
[ -s "$E/eval/teacher-prod.npz" ] || { echo "no reference: run local/secprof/run-eval-sec.sh prod first"; exit 1; }
mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
$PY -c "
import numpy as np
t = np.load('$B/sec_eval.npy').reshape(4, -1)
np.save('$O/tokens.npy', t)
for i, s in enumerate(t): s.astype(np.int32).tofile('$O/seq%d.i32' % i)"
for i in 0 1 2 3; do
  env HIP_VISIBLE_DEVICES=0,1,2 STRATA_PF_NO_DUMMY=1 STRATA_NO_LARGEPAGES=1 STRATA_PEER_GIB=28,20 STRATA_PEER_FILL=file \
    "$BIN" -m "$MODEL" --ids-file "$O/seq$i.i32" -n 1 --temp 0 \
    --backend cuda --experts gpu --arena-gib 40 --arena-adapt --peer-device 1,2 --ctx 16384 --kv-fp4 --threads 12 \
    --rank "${RANK:-/home/yuya/models/dsv41-1017-strata/rank-lucebox.bin}" \
    --engram /home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
    --route-bias 0 --route-bias-file 0 --ppl --dump-logits "$O/seq$i.f32" > "$O/seq$i.out" 2> "$O/seq$i.log"
  echo "seq $i: $(grep '^PPL' "$O/seq$i.out")"
done
$PY $E/score_sec.py score "$O" "$O/tokens.npy" $E/eval/teacher-prod.npz | tee "$O/score.json"
rm -f "$O"/seq*.f32
