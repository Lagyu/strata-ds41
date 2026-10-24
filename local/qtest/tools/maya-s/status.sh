#!/bin/bash
# status.sh - where the Maya-S preparation stands (download, ABL patch, merge, build, parity) and whether the window can run.
D=/mnt/ext/models/maya-s-1017; S=/home/yuya/ds41/maya-s; R=/home/yuya/llm-src/strata-ds41
for u in mayas-dl mayas-merge mayas-build mayas-chain; do printf '%-12s %s\n' "$u" "$(systemctl is-active $u 2>/dev/null)"; done
echo "-- download: $(ls "$D"/*.ok 2>/dev/null | wc -l)/4 shards verified$( [ -e "$D/DOWNLOAD.done" ] && echo ', DOWNLOAD.done')"
tail -c 400 "$S/download.log" 2>/dev/null | tr '\r' '\n' | grep -v '^$' | tail -1
echo "-- ABL: $(cat "$D/ABL.done" 2>/dev/null || echo 'not yet')"
echo "-- merge: $(cat "$D/MERGE.done" 2>/dev/null || tail -1 "$S/merge.log" 2>/dev/null)"
echo "-- build: $(grep -E 'BUILD OK|FAILED' "$S/build.log" 2>/dev/null | tail -1 || echo 'running/not started'); $(tail -1 "$S/build.log" 2>/dev/null | cut -c1-100)"
echo "-- parity: $( [ -e "$R/local/maya-s/parity.ok" ] && tail -1 "$R/local/maya-s/parity.ok" || echo 'not run (the chain runs it on card 0)')"
echo "-- P41 copy: $( [ -e /mnt/win/llm/maya-s/DS41-1017-MayaS-ABL.gguf ] && stat -c '%s bytes' /mnt/win/llm/maya-s/DS41-1017-MayaS-ABL.gguf || echo 'none yet (the chain copies it)')"
df -h /mnt/ext /mnt/win | tail -2
