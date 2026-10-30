#!/bin/bash
# ggml-obj.sh REL_SRC OUT_OBJ - compile one ggml-cuda source of the private ggml tree (/home/yuya/llm-src/zcnvis/llama.cpp) with
# the main build's exact HIP command (compile_commands.json), paths mapped to the private tree.  REL_SRC like ggml-cuda/ds41-fused.cu
set -u
REL=$1 OUT=$2
M=/home/yuya/llm-src/strata-ds41/third_party/llama.cpp; SRC=/home/yuya/llm-src/zcnvis/llama.cpp
python3 -I - "$M" "$SRC" "$REL" "$OUT" <<'PY' > /tmp/ggml-obj-cmd.sh
import json, shlex, sys
M, SRC, REL, OUT = sys.argv[1:5]
for e in json.load(open(M + '/build-hip/compile_commands.json')):
    if e['file'] == M + '/ggml/src/' + REL:
        a = shlex.split(e['command'])
        i = a.index('-o'); a[i + 1] = OUT
        a = [x.replace(M + '/ggml/', SRC + '/ggml/') if '/build-hip/' not in x else x for x in a]
        a.insert(1, '-ffile-prefix-map=' + SRC + '=' + M)
        print('cd ' + shlex.quote(e['directory']) + ' && nice -n 19 ' + ' '.join(shlex.quote(x) for x in a))
        break
else:
    sys.exit('no compile command for ' + REL)
PY
[ -s /tmp/ggml-obj-cmd.sh ] || exit 1
bash /tmp/ggml-obj-cmd.sh
