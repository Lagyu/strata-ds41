#!/bin/bash
# splice-quant.sh [PLAN] - tools/v41/splice_b.py quant for a residency-splice plan (resumable: experts already in
# /mnt/win/llm/splice-b/q are skipped), with the CPU package temperature once a minute.  As a transient unit:
#   sudo systemd-run --unit=splice-quant --property=User=yuya --collect /bin/bash /home/yuya/llm-src/strata-ds41/local/splice-quant.sh PLAN
# Log: logs/splice-quant.log
cd /home/yuya/llm-src/strata-ds41 || exit 1
PLAN=${1:-/mnt/win/llm/splice-b/plan-bs.json}
exec >> logs/splice-quant.log 2>&1
export HOME=/home/yuya
echo "== splice-quant $PLAN start $(date '+%F %T')"
( while :; do echo "temp $(date +%T) $(sensors 2>/dev/null | awk '/Package id 0/{print $4}')"; sleep 60; done ) & TP=$!
/home/yuya/llm-src/hfenv/bin/python tools/v41/splice_b.py quant --plan "$PLAN" \
  --src /mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf \
  --imatrix /home/yuya/models/unsloth-1017/imatrix/imatrix_unsloth.gguf_file \
  --qdir /mnt/win/llm/splice-b/q --threads 28
echo "== splice-quant end rc=$? $(date '+%F %T')"
kill $TP
