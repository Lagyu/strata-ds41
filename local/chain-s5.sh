#!/bin/bash
# After chain-s4: the held-out quality evaluation (chain-s3's attempt ran the system python without numpy), then
# the default configuration back on the service for normal use.
cd /home/yuya/llm-src/strata-ds41
sudo systemctl stop strata-ds41.service
sleep 5
local/eval-lucebox/run-eval.sh mxfp4-abl 2>&1 | tail -14
echo local/ds41-yuya-3gpu.json > local/active-config
sudo systemctl start strata-ds41.service
echo "$(date +%T) chain-s5 done"
