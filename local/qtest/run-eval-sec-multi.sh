#!/usr/bin/env bash
# run-eval-sec-multi.sh TAG MODEL teacher OUT.npz          the security held-out set through the 3-card residency; writes OUT.npz
#                                                          (top-256 log-probs + NLL of THIS model = a reference)
# run-eval-sec-multi.sh TAG MODEL score REF1.npz [REF2...] the same run scored against each reference (KL(ref || TAG),
#                                                          top-1 agreement, both ppl) -> local/secprof/eval/TAG/score-<ref>.json
# Same data (16 x 2048 tokens as 4 sequences of 8192), residency and flags as local/secprof/run-eval-sec.sh; BIN overrides the
# engine binary (default build-hip-mayas/ds41_generate). The chat server must be stopped.
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN=${BIN:-./build-hip-mayas/ds41_generate}
TAG=$1; MODEL=$2; MODE=$3; shift 3
E=local/secprof; O=$E/eval/$TAG; B=/home/yuya/datasets/security-profile/build; PY=/home/yuya/llm-src/hfenv/bin/python
mkdir -p "$O"
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
$PY -c "
import numpy as np
t = np.load('$B/sec_eval.npy').reshape(4, -1)
np.save('$O/tokens.npy', t)
for i, s in enumerate(t): s.astype(np.int32).tofile('$O/seq%d.i32' % i)"
for i in 0 1 2 3; do
  env HIP_VISIBLE_DEVICES=0,1,2 STRATA_PF_NO_DUMMY=1 STRATA_NO_LARGEPAGES=1 STRATA_PEER_GIB=28,20 STRATA_PEER_FILL=file \
    timeout -k 20 "${TMO:-2400}" "$BIN" -m "$MODEL" --ids-file "$O/seq$i.i32" -n 1 --temp 0 \
    --backend cuda --experts gpu --arena-gib 40 --arena-adapt --peer-device 1,2 --ctx 16384 --kv-fp4 --threads 12 \
    --rank "${RANK:-/home/yuya/models/dsv41-1017-strata/rank-lucebox.bin}" \
    --engram /home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf \
    --route-bias 0 --route-bias-file 0 --ppl --dump-logits "$O/seq$i.f32" ${ENGINE_ARGS:-} > "$O/seq$i.out" 2> "$O/seq$i.log"
  echo "seq $i: $(grep '^PPL' "$O/seq$i.out")"
done
if [ "$MODE" = teacher ]; then
  $PY $E/score_sec.py teacher "$O" "$O/tokens.npy" "$1" | tee "$O/score.json"
else
  for ref in "$@"; do
    n=$(basename "$ref" .npz)
    echo "vs $n: $($PY $E/score_sec.py score "$O" "$O/tokens.npy" "$ref" | tee "$O/score-$n.json")"
  done
fi
rm -f "$O"/seq*.f32
