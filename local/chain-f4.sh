#!/bin/bash
# chain-f4 (after chain-f3): rebuild with STRATA_DSPARK_DEVICE, then the drafter on a peer card (its 4.9 GB + head back
# to the primary's expert slots; that peer's store shrunk by 6 GiB), on the R9700 (device 1) and on the second XTX
# (device 2), then a reference on the same binary.
cd /home/yuya/llm-src/strata-ds41
until grep -q "chain-f3 done" logs/chain-f3.log 2>/dev/null; do sleep 30; done
sudo systemctl stop strata-ds41.service
if ! cmake --build build-hip --target ds41_generate -j 24 2>&1 | grep -E "error:|Linking HIP"; then echo "build failed"; sudo systemctl start strata-ds41.service; exit 1; fi
B=local/ds41-yuya-3gpu.json
local/sweep.sh F-dspdev1 "$B" STRATA_DSPARK_DEVICE=1 STRATA_PEER_GIB=22,20
local/sweep.sh F-dspdev2 "$B" STRATA_DSPARK_DEVICE=2 STRATA_PEER_GIB=28,14
local/sweep.sh F-ref3 "$B"
(cd /home/yuya/ds41 && python3 table.py F-ref F-ref2 F-ref3 F-keep96 F-arena100 F-dspdev1 F-dspdev2)
grep -E "DSpark: on|expert slots|VRAM:" local/serve-engine.log | tail -3 | cut -c1-160
echo "$B" > local/active-config
sudo systemctl restart strata-ds41.service
echo "chain-f4 done $(date +%H:%M)"
