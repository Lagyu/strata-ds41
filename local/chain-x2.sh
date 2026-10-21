#!/bin/bash
# chain-x2 (2026-10-21, after chain-x1): the Engram hot set (upstream's pinned-row pack, 4 GiB, built by
# local/engram-hot.sh) measured on the XQ v3 service config as its own sweep row (STRATA_ENGRAM_HOT=PACK); the service
# is left on whichever of the two was faster (the hot set never changes outputs: the rows are the same bytes).
# Log: logs/chain-x2.log.
cd /home/yuya/llm-src/strata-ds41
PACK=/home/yuya/models/dsv41-1017-strata/engram-hot-4g.bin
until grep -q "chain-x1 done" logs/chain-x1.log 2>/dev/null; do sleep 30; done
echo "== XQ v3 + Engram hot set $(date +%H:%M)"
local/sweep.sh X-xq3-hot local/ds41-yuya-3gpu-xq3.json STRATA_ENGRAM_HOT=$PACK
grep -E "engram: hot set|engram decode reads" local/serve-engine.log | tail -3 | cut -c1-160
echo "== MXFP4-ABL + Engram hot set $(date +%H:%M)"
local/sweep.sh F-hot local/ds41-yuya-3gpu.json STRATA_ENGRAM_HOT=$PACK
grep -E "engram: hot set|engram decode reads" local/serve-engine.log | tail -3 | cut -c1-160
(cd /home/yuya/ds41 && python3 table.py F-ref3 F-hot X-xq3 X-xq3-hot)
# the service goes back to the production config (quality first; the hot set is adopted only after the user sees the row)
echo local/ds41-yuya-3gpu.json > local/active-config
sudo systemctl restart strata-ds41.service
until curl -s -m 5 http://127.0.0.1:8150/v1/models | grep -q loaded; do sleep 10; done
echo "chain-x2 done $(date +%H:%M) (service on local/ds41-yuya-3gpu.json)"
