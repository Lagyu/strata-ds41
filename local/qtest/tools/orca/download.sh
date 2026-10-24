#!/bin/bash
# download.sh - orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF, the Mix-256G build's shards 1 and 3-7 (235.8 GB) onto
# the USB SSD, each verified against its published sha256 (HF revision 43fac799, 2026-10-24). Shard 2 = their Engram table
# (IQ4_NL, 110.6 GB) is NOT fetched: the engine keeps --engram on the Lucebox FP8 tables (their own eval used a lossless
# MXFP8 Engram). Rate cap RATE (default 64M = about the Wi-Fi link). Resumable: rerun the same command. Launch:
#   sudo systemd-run --unit=orca-dl --property=User=yuya --property=Nice=10 --property=MemoryMax=12G --collect \
#        /bin/bash /home/yuya/ds41/orca/download.sh
set -u
S=/home/yuya/ds41/orca
D=/mnt/ext/models/orca-mix256
B=https://huggingface.co/orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF/resolve/main/Mix-256G
N=DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G
RATE=${RATE:-64M}
mkdir -p "$D"
exec >> "$S/download.log" 2>&1
echo "$(date '+%F %T') download.sh start (pid $$, cap $RATE)"
declare -A SHA=(
  [1]=2cb066de483877f75b3dc79936b1fccf34aeecc60461701e5c39c98d7f922e03
  [3]=b2583b9da6c3598f43d22714f3216709ab3160c618ad94e36d491a3bb20b5d2b
  [4]=de9c5f8f3462c7df3db7bb2b79301eeab9d7874f581a9fa574ae4fffe127c602
  [5]=1202e0ae311beb38de1b0899ab3cbba485385299a62e8e136e01fa922f0aeb13
  [6]=dc7ce9ee6910e8f46a201e54c96e47b6b3bc9380c88b6187be3c860a0d0b1074
  [7]=ed00667671a8ca5c48da45e022b6482fc2d88a21848a98d26fba1d1f2848b429
)
fetch() {
  local n=$1 f="$N-0000$1-of-00007.gguf" try rc got
  [ -e "$D/$f.ok" ] && { echo "$(date +%T) shard $n: already verified"; return 0; }
  for try in 1 2 3; do
    echo "$(date +%T) shard $n: aria2c (try $try)"
    aria2c -c -x 4 -s 4 -j 1 -k 64M --max-overall-download-limit="$RATE" --file-allocation=none --auto-file-renaming=false \
      --retry-wait=15 --max-tries=0 --timeout=60 --summary-interval=300 --console-log-level=warn \
      -d "$D" -o "$f" "$B/$f"
    rc=$?
    if [ -e "$D/$f.aria2" ]; then echo "$(date +%T) shard $n: incomplete (aria2c exit $rc)"; return 1; fi
    [ -e "$D/$f" ] || { echo "$(date +%T) shard $n: no file (aria2c exit $rc)"; return 1; }
    echo "$(date +%T) shard $n: sha256 ..."
    got=$(sha256sum "$D/$f" | cut -d' ' -f1)
    if [ "$got" = "${SHA[$n]}" ]; then echo "$(date +%T) shard $n: sha256 OK"; touch "$D/$f.ok"; return 0; fi
    echo "$(date +%T) shard $n: sha256 MISMATCH $got - deleting and retrying"
    rm -f "${D:?}/${f:?}"
  done
  return 1
}
for n in 1 3 4 5 6 7; do fetch $n || exit 1; done
touch "$D/DOWNLOAD.done"
echo "$(date '+%F %T') all six shards verified: $D/DOWNLOAD.done"
