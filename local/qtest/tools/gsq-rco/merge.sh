#!/bin/bash
# merge.sh - waits for the download (DOWNLOAD.done), then builds and checks the merged GGUF (merge_experts.py): production's
# metadata + abliterated dense tensors with GSQ-RCO IQ2_XS's routed experts. Launch:
#   sudo systemd-run --unit=gsq-merge --property=User=yuya --property=Nice=15 --property=MemoryMax=6G --collect \
#        /bin/bash /home/yuya/ds41/gsq-rco/merge.sh
set -u
S=/home/yuya/ds41/gsq-rco; D=/mnt/ext/models/gsq-rco-1017
SRC=/mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf
# 2026-10-24 11:25 (user: the server is idle, start anytime): written straight to the P41 where the chain reads it (no copy
# in the window); the marker MERGE.done stays in $D
OUT=/mnt/win/llm/qtest/DS41-1017-GSQRCO-IQ2XS-ABL.gguf
mkdir -p /mnt/win/llm/qtest
PY=/home/yuya/llm-src/hfenv/bin/python
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
ARGS=(--src "$SRC" --shards "$D/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-0000?-of-00004.gguf" --out "$OUT"
      --name "DeepSeek V4.1 Flash 1017 (GSQ-RCO IQ2_XS experts, MXFP4-ABL dense)"
      --note "GSQ-RCO IQ2_XS routed experts (ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF: per-layer IQ2_XXS / IQ2_XS / IQ2_S gate=up, IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS down); Q8_0 attention/shared/head with the abliteration, F16 embedding, from the MXFP4-ABL file"
      --source "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF IQ2_XS shards 1-4, HF revision cb02422c (2026-10-23)")
exec >> "$S/merge.log" 2>&1
echo "$(date '+%F %T') merge.sh start (pid $$)"
[ -e "$D/MERGE.done" ] && { echo "already merged: $(cat "$D/MERGE.done")"; exit 0; }
until [ -e "$D/DOWNLOAD.done" ]; do sleep 60; done
echo "$(date '+%F %T') download complete; building $OUT"
rm -f "${OUT:?}"
"$PY" -I "$S/merge_experts.py" build "${ARGS[@]}" || { echo "$(date '+%F %T') BUILD FAILED"; exit 1; }
echo "$(date '+%F %T') check"
"$PY" -I "$S/merge_experts.py" check "${ARGS[@]}" --per-layer 6 || { echo "$(date '+%F %T') CHECK FAILED"; exit 1; }
ls -l "$OUT" > "$D/MERGE.done"
echo "$(date '+%F %T') MERGE.done: $(cat "$D/MERGE.done")"
