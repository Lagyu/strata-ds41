#!/bin/bash
# deploy-crit.sh [KEY=VALUE ...] - production gets the critpath window's binary (default ds41_generate-c3: production code + the
# RDNA matrix-core lightning indexer + fused hc kernels in its ggml archive, the hinted file-tier reads, skip-read, seed buffers,
# the two-drive / partial chunk-prefetch slab; new switches inert unless set, except STRATA_HC_MIXES_FUSED and STRATA_CHUNK_SKIP_READ
# which default ON) and the given env KEY=VALUE additions (e.g. STRATA_INDEXER_WMMA=1 STRATA_CHUNK_HINT_GIB=4
# STRATA_CHUNK_PF_MIN=512 STRATA_CHUNK_PF_FRAC=0.5).  Rollback: local/ds41-yuya-3gpu.json.prod-1030d +
# build-hip/ds41_generate.prod-1030d + /home/yuya/dsh-work/tools/restart-ds41.sh.
set -u
R=/home/yuya/llm-src/strata-ds41; CFG=$R/local/ds41-yuya-3gpu.json
BIN_NEW=${BIN_NEW:-/home/yuya/llm-src/strata-zcnvis/build-hip-deploy/ds41_generate-c3}
NOTE=${NOTE:-}
ARGS=${ARGS:-}   # extra engine args appended to the config's args, e.g. "--chunk-prefetch 4"
cd "$R" || exit 1
[ -f "$CFG.prod-1030d" ] && { echo "rollback copy $CFG.prod-1030d exists - already deployed?"; exit 1; }
[ -x "$BIN_NEW" ] || { echo "no binary at $BIN_NEW"; exit 1; }
strings -n 12 "$BIN_NEW" | grep -q 'STRATA_INDEXER_WMMA' || { echo "$BIN_NEW has no matrix-core indexer"; exit 1; }
cp -a "$CFG" "$CFG.prod-1030d"
python3 -I - "$CFG" "$NOTE" "$ARGS" "$@" <<'PY' || { cp -a "$CFG.prod-1030d" "$CFG"; echo "config edit failed - restored"; exit 1; }
import json, re, sys
p, note, args_add = sys.argv[1], sys.argv[2], sys.argv[3]
extra = dict(kv.split('=', 1) for kv in sys.argv[4:])
s = open(p).read()
if args_add.strip():
    last_arg = '  "0.10"\n ],'; assert s.count(last_arg) == 1, 'last arg line'
    s = s.replace(last_arg, '  "0.10"' + ''.join(',\n  "%s"' % a for a in args_add.split()) + '\n ],')
last = '  "STRATA_D3_DMA_STREAM": "1"\n'; assert s.count(last) == 1, 'last env line'
add = ''.join(',\n  "%s": "%s"' % (k, v) for k, v in extra.items())
s = s.replace(last, '  "STRATA_D3_DMA_STREAM": "1"' + add + '\n')
if note:
    m = re.search(r'^ "_comment": "(.*)",$', s, re.M); assert m, '_comment'
    s = s[:m.end(1)] + ' ' + note.replace('"', "'") + s[m.end(1):]
c = json.loads(s)
for k, v in extra.items(): assert c['env'][k] == v
for a in args_add.split(): assert a in c['args']
open(p, 'w').write(s)
print('config edited:', ', '.join('%s=%s' % kv for kv in extra.items()) or '(no env change)', '| args +', args_add or '(none)')
PY
cp -a build-hip/ds41_generate build-hip/ds41_generate.prod-1030d
# the running engine holds build-hip/ds41_generate open: copy beside it and rename over it
cp -a "$BIN_NEW" build-hip/ds41_generate.new && mv -f build-hip/ds41_generate.new build-hip/ds41_generate && cmp -s "$BIN_NEW" build-hip/ds41_generate \
  || { echo "binary install failed"; cp -a build-hip/ds41_generate.prod-1030d build-hip/ds41_generate.new && mv -f build-hip/ds41_generate.new build-hip/ds41_generate; cp -a "$CFG.prod-1030d" "$CFG"; exit 1; }
echo "binary: $(sha256sum build-hip/ds41_generate | cut -c1-16) (rollback build-hip/ds41_generate.prod-1030d)"
bash /home/yuya/dsh-work/tools/restart-ds41.sh || exit 1
K=$(grep -E '^STRATA_API_KEY=' /home/yuya/.config/strata-ds41/lan.env 2>/dev/null | cut -d= -f2-)
ans=$(curl -s -m 400 -H "Authorization: Bearer $K" -H 'Content-Type: application/json' http://127.0.0.1:8150/v1/chat/completions \
  -d '{"model":"deepseek-v4.1-flash-1017","messages":[{"role":"user","content":"What is 17 + 25? Answer with the number only."}],"max_tokens":200,"temperature":0}' \
  | python3 -c 'import json,sys; print((json.load(sys.stdin)["choices"][0]["message"].get("content") or "").strip())' 2>/dev/null)
P=$(pgrep -x ds41_generate | head -1)
echo "smoke answer '$ans'; engine $(sha256sum /proc/$P/exe 2>/dev/null | cut -c1-16); env: $(tr '\0' '\n' < /proc/$P/environ 2>/dev/null | grep -E '^(STRATA_INDEXER_WMMA|STRATA_CHUNK_HINT_GIB|STRATA_HC_MIXES_FUSED|STRATA_CHUNK_SKIP_READ|STRATA_CHUNK_PF_MIN|STRATA_CHUNK_PF_FRAC|STRATA_ENGRAM_HOT|STRATA_CHUNK_ASYNC)=' | tr '\n' ' ')"
grep -aE '^(indexer:|chunk prefetch:|chunk hint:|tier:)' local/serve-engine.log | tail -3 | cut -c1-200
