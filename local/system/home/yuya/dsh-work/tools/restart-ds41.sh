#!/bin/bash
# Restart strata-ds41: stop, wait until every GPU's VRAM is released (a pinned 95 GiB arena takes a while to tear
# down; a start before that fails with "cannot allocate the dense weights"), start, wait for the API (with its key).
set -u
K=$(grep -E '^STRATA_API_KEY=' /home/yuya/.config/strata-ds41/lan.env 2>/dev/null | cut -d= -f2-)
sudo systemctl stop strata-ds41.service
for i in $(seq 1 60); do
  busy=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk -F': ' '{s+=$NF} END{print s+0}')
  [ "$busy" = "0" ] && break
  sleep 2
done
echo "VRAM released after ~$((i * 2)) s (sum of VRAM%: $busy)"
sudo systemctl start strata-ds41.service || exit 1
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $K" http://127.0.0.1:8150/v1/models 2>/dev/null)
  [ "$code" = "200" ] && break
  systemctl is-active -q strata-ds41.service || { echo "the service died while loading"; tail -5 /home/yuya/llm-src/strata-ds41/local/serve-engine.log | cut -c1-200; exit 1; }
  sleep 5
done
[ "$code" = "200" ] && echo "ready after ~$((i * 5)) s" || { echo "not ready after $((i * 5)) s"; exit 1; }
