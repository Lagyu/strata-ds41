#!/bin/bash
# run-profile.sh - start the service on local/variants/secprof.json (STRATA_ROUTE_COUNTS on), drive it through the
# security corpora (local/secprof/profile.py), wait for the counts' last periodic write, then stop the service and put
# local/active-config back to production.  As a transient unit:
#   sudo systemd-run --unit=secprof --property=User=yuya --collect /bin/bash /home/yuya/llm-src/strata-ds41/local/secprof/run-profile.sh
# Log: logs/secprof.log
cd /home/yuya/llm-src/strata-ds41 || exit 1
exec >> logs/secprof.log 2>&1
export PATH=$PATH:/opt/rocm/bin HOME=/home/yuya
B=/home/yuya/datasets/security-profile/build
echo "== secprof start $(date '+%F %T')"
echo local/variants/secprof.json > local/active-config
sudo systemctl stop strata-ds41.service
for i in $(seq 1 30); do
  busy=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk -F': ' '{s+=$NF} END{print s+0}')
  [ "$busy" = "0" ] && break
  sleep 2
done
sudo systemctl start strata-ds41.service || exit 1
for i in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8150/v1/models)" = "200" ] && break
  sleep 5
done
echo "service ready $(date +%T)"
/home/yuya/llm-src/hfenv/bin/python local/secprof/profile.py $B --max-requests 80 --gen 6
# one more short request after 35 s so the counter's periodic write covers everything above
sleep 35
curl -s -m 600 http://127.0.0.1:8150/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash-1017","messages":[{"role":"user","content":"ok"}],"max_tokens":1,"reasoning_effort":"none"}' > /dev/null
sleep 5
cp $B/route-counts.csv $B/route-counts.final.csv
sudo systemctl stop strata-ds41.service
echo local/ds41-yuya-3gpu.json > local/active-config
echo "== secprof end $(date '+%F %T')"
