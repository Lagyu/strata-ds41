#!/bin/bash
# download.sh - Maya-S (peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF) model files 1-4 of 6 onto the USB SSD, file 4
# first, each verified against its published sha256, then the abliteration written into file 4 (abl-patch-shard4.sh).
# Files 5-6 (Engram tables, Q6_K) and dspark/ are NOT needed: our engine keeps --engram on the Lucebox FP8 tables and
# its own drafter. Rate-capped at 32 MiB/s (Wi-Fi ~43 MB/s; the VPN shares the link). Resumable: rerun the same command.
# Launch (survives harness restarts):
#   sudo systemd-run --unit=mayas-dl --property=User=yuya --property=Nice=10 --property=MemoryMax=12G --collect \
#        /bin/bash /home/yuya/ds41/maya-s/download.sh
set -u
S=/home/yuya/ds41/maya-s
D=/mnt/ext/models/maya-s-1017
B=https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF/resolve/main/Maya-S
N=DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS
LOG=$S/download.log
mkdir -p "$D"
exec >> "$LOG" 2>&1
echo "$(date '+%F %T') download.sh start (pid $$)"
declare -A SHA=(
  [1]=1f3437f57042e69f94c141ffbcba0d5610dab63feb07244fb468be4940d6763f
  [2]=66501fee8702ffd54e6f065323ab54bb3150d20b10b01922c589e7462c7b5ba6
  [3]=3195b23f33337c46a07c48583946d99147c3635412bc4adc6acdc4cf42bf30ad
  [4]=d60dba35fac9601cd5c6026439dd980c457a5bb1c009192ea539b220108ea0db
)
fetch() {   # fetch N: download + verify shard N (skipped when $D/<file>.ok exists)
  local n=$1 f="$N-0000$1-of-00006.gguf" try rc got
  [ -e "$D/$f.ok" ] && { echo "$(date +%T) shard $n: already verified"; return 0; }
  for try in 1 2 3; do
    echo "$(date +%T) shard $n: aria2c (try $try)"
    aria2c -c -x 4 -s 4 -j 1 -k 64M --max-overall-download-limit=32M --file-allocation=none --auto-file-renaming=false \
      --retry-wait=15 --max-tries=0 --timeout=60 --summary-interval=300 --console-log-level=warn \
      -d "$D" -o "$f" "$B/$f"
    rc=$?
    if [ -e "$D/$f.aria2" ]; then echo "$(date +%T) shard $n: incomplete (aria2c exit $rc)"; return 1; fi
    [ -e "$D/$f" ] || { echo "$(date +%T) shard $n: no file (aria2c exit $rc)"; return 1; }
    echo "$(date +%T) shard $n: sha256 ..."
    got=$(sha256sum "$D/$f" | cut -d' ' -f1)
    if [ "$got" = "${SHA[$n]}" ]; then echo "$(date +%T) shard $n: sha256 OK"; touch "$D/$f.ok"; return 0; fi
    echo "$(date +%T) shard $n: sha256 MISMATCH $got - deleting and retrying"
    rm -f "$D/$f"
  done
  return 1
}
fetch 4 || exit 1
bash "$S/abl-patch-shard4.sh" || echo "$(date +%T) ABL patch FAILED (rerun abl-patch-shard4.sh by hand); continuing with shards 1-3"
for n in 1 2 3; do fetch $n || exit 1; done
touch "$D/DOWNLOAD.done"
echo "$(date '+%F %T') all four shards verified: $D/DOWNLOAD.done"
