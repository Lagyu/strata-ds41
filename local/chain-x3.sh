#!/bin/bash
# chain-x3 (2026-10-21 16:00): the rows the session restart cut short - the Engram hot set on the production config
# (F-hot) and a clean post-hardware-change reference (F-ref4: CPU re-greased, memory timing set, P41 on the CPU M.2)
# - then the production config back.  Run as a systemd transient unit (survives Claude Code session restarts):
#   sudo systemd-run --unit=ds41-chain-x3 --property=User=yuya --collect /bin/bash /home/yuya/llm-src/strata-ds41/local/chain-x3.sh
# Log: logs/chain-x3.log.
cd /home/yuya/llm-src/strata-ds41 || exit 1
exec >> /home/yuya/llm-src/strata-ds41/logs/chain-x3.log 2>&1
export PATH=$PATH:/opt/rocm/bin:/usr/local/bin
export HOME=/home/yuya
PACK=/home/yuya/models/dsv41-1017-strata/engram-hot-4g.bin
echo "== chain-x3 start $(date +%H:%M)"
echo "== MXFP4-ABL + Engram hot set $(date +%H:%M)"
local/sweep.sh F-hot local/ds41-yuya-3gpu.json STRATA_ENGRAM_HOT=$PACK
grep -E "engram: hot set|engram decode reads" local/serve-engine.log | tail -3 | cut -c1-160
echo "== clean post-hardware reference $(date +%H:%M)"
local/sweep.sh F-ref4 local/ds41-yuya-3gpu.json
(cd /home/yuya/ds41 && python3 table.py F-ref F-ref3 F-ref4 F-hot X-xq3 X-xq3-hot)
echo local/ds41-yuya-3gpu.json > local/active-config
sudo systemctl restart strata-ds41.service
until curl -s -m 5 http://127.0.0.1:8150/v1/models | grep -q loaded; do sleep 10; done
echo "chain-x3 done $(date +%H:%M) (service on local/ds41-yuya-3gpu.json)"
