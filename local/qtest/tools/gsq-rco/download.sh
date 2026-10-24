#!/bin/bash
# download.sh - ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF, the IQ2_XS build's 4 shards (155.2 GB) onto the USB SSD,
# each verified against its published sha256 (HF revision cb02422c, 2026-10-23). The shared Engram file (IQ4_NL, 110.6 GB)
# is NOT fetched: the engine keeps --engram on the Lucebox FP8 tables, as for Maya-S. No in-place ABL: this build's
# attn_output_b / ffn_down_shexp are Q5_K (RCO's dense ladder Q5_K/Q6_K/Q8_0), so the edit cannot be written into them
# exactly; the merged file takes production's abliterated Q8_0 dense instead (merge.sh). Rate-capped at 32 MiB/s.
# Resumable: rerun the same command. Launch:
#   sudo systemd-run --unit=gsq-dl --property=User=yuya --property=Nice=10 --property=MemoryMax=12G --collect \
#        /bin/bash /home/yuya/ds41/gsq-rco/download.sh
set -u
S=/home/yuya/ds41/gsq-rco
D=/mnt/ext/models/gsq-rco-1017
B=https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/main
N=DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS
mkdir -p "$D"
exec >> "$S/download.log" 2>&1
echo "$(date '+%F %T') download.sh start (pid $$)"
declare -A SHA=(
  [1]=4c65307992ab586c527ca1f6f64062aa96a33b23c02b7fed103f45f75e854798
  [2]=933461a974fcccee741bf2f9798201e72ab4aa3f8a952be56f849896ca6ea30e
  [3]=bbf91fa346a85ebaf32dd4fd5c456b534d780d381e54f46ac90783a481373240
  [4]=e5b2f8781fab688cd4bc3bc7a69651b5dbb90ed94965657c6bead8bb4774f3cf
)
fetch() {
  local n=$1 f="$N-0000$1-of-00004.gguf" try rc got
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
    rm -f "${D:?}/${f:?}"
  done
  return 1
}
for n in 1 2 3 4; do fetch $n || exit 1; done
touch "$D/DOWNLOAD.done"
echo "$(date '+%F %T') all four shards verified: $D/DOWNLOAD.done"
