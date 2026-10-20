#!/bin/bash
# the service up on the default config, then chain-u1 (unsloth Q8/Q4 bench + evals) and chain-u2 (U-q8 rerun, REBAL, turnbench)
cd /home/yuya/llm-src/strata-ds41
echo local/ds41-yuya-3gpu.json > local/active-config
sudo systemctl start strata-ds41.service
for i in $(seq 1 90); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8150/v1/models 2>/dev/null | grep -q 200 && break; sleep 5; done
echo "$(date +%T) service: $(systemctl is-active strata-ds41.service), api $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8150/v1/models)"
bash local/chain-u1.sh > logs/chain-u1.log 2>&1
bash local/chain-u2.sh > logs/chain-u2.log 2>&1
echo "$(date +%T) run-u1u2 done"
