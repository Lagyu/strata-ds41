#!/bin/bash
# chain-l2 (after chain-l1): a needle test at the model's maximum context (1,040,000 tokens + the question) on the 1M
# variant, then the service back on the default config.
cd /home/yuya/llm-src/strata-ds41
PY=/home/yuya/llm-src/hfenv/bin/python3
until grep -q "chain-l1 done" logs/chain-l1.log 2>/dev/null; do sleep 60; done
until curl -s -m 5 http://127.0.0.1:8150/v1/models | grep -q loaded; do sleep 10; done
echo "== NIAH 1M $(date +%H:%M)"
$PY local/longctx/niah_run.py --tokens 1040000 --depths 0.5 --label niah-1m
echo local/ds41-yuya-3gpu.json > local/active-config
sudo systemctl restart strata-ds41.service
echo "chain-l2 done $(date +%H:%M)"
