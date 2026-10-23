#!/bin/bash
# parity-check.sh TAG - run-eval-3gpu.sh TAG, then compare its score with base-3gpu digit for digit and list kernel
# errors seen during the run.  As a transient unit (survives Claude Code session restarts):
#   sudo systemd-run --unit=ds41-parity --property=User=yuya --property=LimitMEMLOCK=infinity \
#     --property=LimitNOFILE=65536 --collect /bin/bash /home/yuya/llm-src/strata-ds41/local/parity-check.sh TAG
# Log: logs/parity-TAG.log
cd /home/yuya/llm-src/strata-ds41 || exit 1
TAG=${1:?usage: parity-check.sh TAG}
exec > logs/parity-$TAG.log 2>&1
export PATH=$PATH:/opt/rocm/bin:/usr/local/bin HOME=/home/yuya
T0=$(date '+%F %T')
echo "== parity $TAG start $T0"
bash local/eval-lucebox/run-eval-3gpu.sh "$TAG" || echo "== parity $TAG: eval FAILED $(date +%T)"
if [ -f local/eval-lucebox/$TAG/score.json ]; then
  /home/yuya/llm-src/hfenv/bin/python - local/eval-lucebox/base-3gpu/score.json local/eval-lucebox/$TAG/score.json <<'PY'
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
print("base-3gpu", a)
print(sys.argv[2].split("/")[-2], b)
print("PARITY OK" if a == b else "PARITY MISMATCH")
PY
fi
echo "kernel errors since the start: $(journalctl -k --since "$T0" --no-pager | grep -cE 'Bad page|BUG:|segfault|Machine check|Hardware Error|general protection')"
echo "== parity $TAG end $(date '+%F %T')"
