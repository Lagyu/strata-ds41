#!/bin/bash
# merge.sh - waits for the download (DOWNLOAD.done), then builds and checks the merged GGUF (merge_mayas.py):
# production's metadata + dense tensors (abliterated) with Maya-S's routed experts. Launch:
#   sudo systemd-run --unit=mayas-merge --property=User=yuya --property=Nice=15 --property=MemoryMax=6G --collect \
#        /bin/bash /home/yuya/ds41/maya-s/merge.sh
set -u
S=/home/yuya/ds41/maya-s; D=/mnt/ext/models/maya-s-1017
SRC=/mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf
OUT=$D/DS41-1017-MayaS-ABL.gguf
PY=/home/yuya/llm-src/hfenv/bin/python
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec >> "$S/merge.log" 2>&1
echo "$(date '+%F %T') merge.sh start (pid $$)"
[ -e "$D/MERGE.done" ] && { echo "already merged: $(cat "$D/MERGE.done")"; exit 0; }
until [ -e "$D/DOWNLOAD.done" ]; do sleep 60; done
echo "$(date '+%F %T') download complete; building $OUT"
rm -f "$OUT"
"$PY" -I "$S/merge_mayas.py" build --src "$SRC" --shards "$D" --out "$OUT" || { echo "$(date '+%F %T') BUILD FAILED"; exit 1; }
echo "$(date '+%F %T') check"
"$PY" -I "$S/merge_mayas.py" check --src "$SRC" --shards "$D" --out "$OUT" --per-layer 6 || { echo "$(date '+%F %T') CHECK FAILED"; exit 1; }
ls -l "$OUT" > "$D/MERGE.done"
echo "$(date '+%F %T') MERGE.done: $(cat "$D/MERGE.done")"
