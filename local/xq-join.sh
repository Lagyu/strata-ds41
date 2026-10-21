#!/bin/bash
# xq-join.sh [DIR] - join IndrasMirror's V41-XQ-1017-v3.gguf from its 16 byte-range parts (upstream's
# tools/v41/hf-1017/reassemble.sh, done here without running the downloaded script): every part is checksummed
# against SHA256SUMS first, the parts are appended in order, the joined file is checksummed, and only then are the
# parts deleted.  Writes DIR/JOINED.ok on success (the chains wait for it).
set -euo pipefail
dir=${1:-/mnt/win/llm/xq-1017-v3}
cd "$dir"
out=V41-XQ-1017-v3.gguf
[ -f SHA256SUMS ] || { echo "SHA256SUMS not found in $dir"; exit 1; }
mapfile -t parts < <(ls -1 "$out".part-*-of-* 2>/dev/null | grep -v '\.aria2$' | sort)
total="${parts[0]##*-of-}"
[ ${#parts[@]} -eq "$((10#$total))" ] || { echo "found ${#parts[@]} parts, expected $total"; exit 1; }
ls "$out".part-*.aria2 >/dev/null 2>&1 && { echo "a download is still in progress (.aria2 control files)"; exit 1; }
echo "checking ${#parts[@]} parts... $(date +%H:%M)"
grep -E "  $out\.part-" SHA256SUMS | sha256sum -c --quiet - || { echo "a part is corrupt - re-download it"; exit 1; }
[ -e "$out" ] && { echo "$out already exists - move it away first"; exit 1; }
for p in "${parts[@]}"; do
    echo "appending $p $(date +%H:%M)"
    cat "$p" >> "$out"
done
echo "verifying $out... $(date +%H:%M)"
if grep -qE "  $out\$" SHA256SUMS; then
    grep -E "  $out\$" SHA256SUMS | sha256sum -c - || { echo "the joined file's checksum FAILED"; exit 1; }
else
    echo "(no checksum for the joined file in SHA256SUMS; the parts verified)"
fi
rm -f "${parts[@]}"
ls -la "$out" rank-1017-v3.bin "$out".experts.json
touch JOINED.ok
echo "OK: $out joined and verified $(date +%H:%M)"
