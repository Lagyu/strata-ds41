#!/bin/bash
# abl-patch-shard4.sh - write our abliteration (mrfakename's rank-2 edit of attn.wo_b + ffn.shared_experts.w2, applied to
# DeepSeek's original 1017 weights in float64 and quantized to Q8_0 by ggml, exactly as in the production file) into
# Maya-S file 4 of 6, which holds all 80 target tensors byte-identical to the Lucebox originals (verified 2026-10-23).
# Backup: $D/abliteration-backup-shard4.bin (+ .json); `patch_gguf.py FILE LIB restore BACKUP` puts the originals back.
set -u
S=/home/yuya/ds41/maya-s; D=/mnt/ext/models/maya-s-1017
F=$D/DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-00004-of-00006.gguf
LIB=/home/yuya/llm-src/lucebox/server/build-hip/deps/llama.cpp/ggml/src/libggml-base.so
PY=/home/yuya/llm-src/hfenv/bin/python
TOOL=/home/yuya/llm-src/ds41-abliterate/patch_gguf.py
A=/home/yuya/models/abliteration
BK=$D/abliteration-backup-shard4.bin
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
[ -e "$F.ok" ] || { echo "$(date +%T) shard 4 not verified yet"; exit 1; }
if [ -e "$D/ABL.done" ]; then echo "$(date +%T) ABL already applied ($(cat "$D/ABL.done"))"; exit 0; fi
if [ -e "$BK" ]; then
  if [ -e "$BK.json" ]; then
    echo "$(date +%T) a previous attempt left a backup: restoring the originals first"
    "$PY" -I "$TOOL" "$F" "$LIB" restore "$BK" || exit 1
  else
    echo "$(date +%T) a previous attempt died mid-way (backup without .json): shard 4 must be re-downloaded"
    rm -f "$F" "$F.ok" "$BK"; exit 1
  fi
  rm -f "$BK" "$BK.json"
fi
echo "$(date +%T) verify: the original 1017 tensors convert to shard 4's exact bytes (all 80)"
"$PY" -I "$TOOL" "$F" "$LIB" verify "$A/original-1017-wo_b-shexp_w2.safetensors" || exit 1
echo "$(date +%T) apply the abliteration edit"
"$PY" -I "$TOOL" "$F" "$LIB" apply-edit "$A/original-1017-wo_b-shexp_w2.safetensors" "$A/mrfakename-1017/abliteration-edit.npz" "$BK" || exit 1
echo "$(date +%T) sha256 of the patched shard 4 ..."
sha256sum "$F" | tee "$F.abl.sha256"
echo "patched $(date '+%F %T'), sha256 $(cut -d' ' -f1 "$F.abl.sha256")" > "$D/ABL.done"
echo "$(date +%T) ABL done"
