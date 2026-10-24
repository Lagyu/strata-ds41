#!/bin/bash
# convert.sh - waits for the Mix-256G download (DOWNLOAD.done), then writes the Lucebox-layout file straight to the P41 and
# checks it (convert_orca.py build / check). Launch:
#   sudo systemd-run --unit=orca-convert --property=User=yuya --property=Nice=15 --property=MemoryMax=8G --collect \
#        /bin/bash /home/yuya/ds41/orca/convert.sh
set -u
S=/home/yuya/ds41/orca; D=/mnt/ext/models/orca-mix256
SRC=/mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf
OUT=/mnt/win/llm/orca/DS41-1017-Mix256G.gguf
PY=/home/yuya/llm-src/hfenv/bin/python
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
ARGS=(--src "$SRC" --shards "$D/DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-0000?-of-00007.gguf" --out "$OUT" --parts 3)
exec >> "$S/convert.log" 2>&1
echo "$(date '+%F %T') convert.sh start (pid $$)"
[ -e "$D/MERGE.done" ] && { echo "already converted: $(cat "$D/MERGE.done")"; exit 0; }
until [ -e "$D/DOWNLOAD.done" ]; do sleep 60; done
echo "$(date '+%F %T') download complete; free on the P41: $(df -h /mnt/win | tail -1 | awk '{print $4}')"
mkdir -p /mnt/win/llm/orca
# a failed earlier attempt may have left shard files; the NTFS driver cannot pre-size a 238 GB file, hence 3 shards of ~80 GB
for k in 1 2 3; do rm -f "/mnt/win/llm/orca/DS41-1017-Mix256G-0000${k}-of-00003.gguf"; done
"$PY" -I "$S/convert_orca.py" build "${ARGS[@]}" || { echo "$(date '+%F %T') BUILD FAILED"; exit 1; }
echo "$(date '+%F %T') check"
"$PY" -I "$S/convert_orca.py" check "${ARGS[@]}" --per-layer 4 || { echo "$(date '+%F %T') CHECK FAILED"; exit 1; }
ls -l /mnt/win/llm/orca/DS41-1017-Mix256G-0000?-of-00003.gguf > "$D/MERGE.done"
echo "$(date '+%F %T') MERGE.done: $(cat "$D/MERGE.done")"
