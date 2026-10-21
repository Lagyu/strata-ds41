#!/bin/bash
# chain-f3 (after chain-f2): the pinned RAM arena one notch larger (95 -> 100 GiB: ~280 more experts out of the NVMe
# tier; RAM headroom is ~23 GiB with swap already touched, so this is the only step tried), then the default restored.
cd /home/yuya/llm-src/strata-ds41
until grep -q "chain-f2 done" logs/chain-f2.log 2>/dev/null; do sleep 30; done
B=local/ds41-yuya-3gpu.json
local/sweep.sh F-arena100 "$B" --arena-gib=100
(cd /home/yuya/ds41 && python3 table.py S-cap8 F-ref F-ref2 F-keep96 F-ctx32k F-arena100)
echo "$B" > local/active-config
sudo systemctl restart strata-ds41.service
echo "chain-f3 done $(date +%H:%M)"
