#!/bin/bash
# local/warmup.sh - after the service starts: wait for the API, then one short request, so the VRAM expert cache and
# the peer stores open now (the engine opens them at the first prompt: ~20 s added to the first answer otherwise).
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8150/v1/models 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 5
done
[ "$code" = "200" ] || exit 0
curl -s -o /dev/null --max-time 300 http://127.0.0.1:8150/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash-1017","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"reasoning_effort":"none"}'
echo "warmup: done ($(date +%T))"
