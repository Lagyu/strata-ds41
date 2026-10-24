#!/bin/bash
# chain-qtest.sh ENVFILE - the measurement window for a candidate model, generalized from local/maya-s/chain-mayas.sh.
# 1. waits for the candidate's preparation (OKMARK; gives up at BADMARK), then for an IDLE server: the engine log
#    (local/serve-engine.log, written only while requests run) untouched for IDLE_MIN minutes and every GPU <= 10 % busy;
# 2. stops production (~45-60 min) and restores it by an EXIT trap whatever happens (only if it was running before);
# 3. copies SRC to the P41 (DST); held-out eval (TAG, vs base-3gpu); security set (TAG, vs production's reference); the
#    speed set on CFG (sweep LABEL) and the refusal check on the candidate; production back; a production speed row
#    (PROD) and production's refusal check; report logs/qtest-NAME-report.txt. Each step skips when its result exists.
# Launch (survives harness restarts):
#   sudo systemd-run --unit=qtest-gsq --property=User=yuya --property=LimitMEMLOCK=infinity --property=LimitNOFILE=65536 \
#        --collect /bin/bash /home/yuya/llm-src/strata-ds41/local/qtest/chain-qtest.sh local/qtest/gsq-iq2xs.env
set -u
R=/home/yuya/llm-src/strata-ds41
cd "$R" || exit 1
. "$1"   # NAME SRC DST CFG TAG LABEL PROD BIN OKMARK BADMARK IDLE_MIN
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/yuya
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p logs
LOG=logs/qtest-$NAME.log; REP=logs/qtest-$NAME-report.txt
exec >> "$LOG" 2>&1
echo "== chain-qtest $NAME start $(date '+%F %T')"
LAN=/home/yuya/.config/strata-ds41/lan.env
[ -f "$LAN" ] && { set -a; . "$LAN"; set +a; }
until [ -e "$OKMARK" ]; do
  [ -e "$BADMARK" ] && { echo "$(date +%T) preparation FAILED ($(head -1 "$BADMARK")) - no window"; exit 1; }
  sleep 60
done
echo "$(date +%T) preparation OK: $(head -1 "$OKMARK")"
[ -x "$BIN" ] && [ -s "$SRC" ] || { echo "missing $BIN or $SRC"; exit 1; }
EL=local/serve-engine.log; note=0
while true; do
  now=$(date +%s); age=$(( (now - $(stat -c %Y "$EL")) / 60 ))
  busy=$(rocm-smi --showuse 2>/dev/null | grep "GPU use" | grep -oE "[0-9]+$" | sort -n | tail -1)
  if [ "$age" -ge "${IDLE_MIN:-30}" ] && [ "${busy:-100}" -le 10 ]; then
    echo "$(date +%T) idle: the engine log untouched for $age min, max GPU use ${busy}%"; break
  fi
  if [ $((now - note)) -ge 1800 ]; then echo "$(date +%T) waiting for an idle server (engine log age $age min, max GPU use ${busy:-?}%)"; note=$now; fi
  sleep 60
done
wait_vram() {
  for i in $(seq 1 90); do
    b=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk -F': ' '{s+=$NF} END{print s+0}')
    [ "$b" = "0" ] && break
    sleep 2
  done
  echo "$(date +%T) VRAM released after ~$((i * 2)) s"
}
WAS=$(systemctl is-active strata-ds41.service)
restore() {
  echo "$(date +%T) restoring production (local/ds41-yuya-3gpu.json)"
  echo local/ds41-yuya-3gpu.json > local/active-config
  if [ "$WAS" = active ]; then bash /home/yuya/dsh-work/tools/restart-ds41.sh
  else sudo systemctl stop strata-ds41.service; echo "production was not running before the window: left stopped"; fi
}
trap restore EXIT
echo "$(date +%T) stopping strata-ds41 (was $WAS)"
sudo systemctl stop strata-ds41.service; wait_vram
if [ ! -e "$DST" ] || [ "$(stat -c %s "$DST")" != "$(stat -c %s "$SRC")" ]; then
  echo "$(date +%T) copying $(stat -c %s "$SRC") bytes to $DST"
  mkdir -p "$(dirname "$DST")"; rm -f "${DST:?}"
  cp "$SRC" "$DST" || { echo "copy failed"; exit 1; }
  echo "$(date +%T) copy done"
fi
if [ ! -s "local/eval-lucebox/$TAG/score.json" ]; then
  echo "$(date +%T) held-out eval"
  BIN="$BIN" MODEL="$DST" bash local/maya-s/run-eval-mayas.sh "$TAG" 2>&1 | tail -9
fi
echo "$TAG: $(cat "local/eval-lucebox/$TAG/score.json" 2>/dev/null)"
echo "base-3gpu: $(cat local/eval-lucebox/base-3gpu/score.json)"
if [ ! -s "local/secprof/eval/$TAG/score.json" ]; then
  echo "$(date +%T) security eval"
  BIN="$BIN" bash local/maya-s/run-eval-sec-mayas.sh "$TAG" "$DST" 2>&1 | tail -5
fi
echo "security $TAG: $(cat "local/secprof/eval/$TAG/score.json" 2>/dev/null)"
if ! grep -q "\"label\": \"$LABEL\"" /home/yuya/ds41/results.jsonl 2>/dev/null; then
  echo "$(date +%T) speed set on $CFG"
  bash local/sweep.sh "$LABEL" "$CFG" 2>&1 | tail -12
fi
if [ ! -s "logs/qtest-$NAME-refusal.txt" ] && grep -q "$(basename "$CFG" .json)\|$LABEL" local/active-config; then
  echo "$(date +%T) refusal check on the candidate (thinking off)"
  python3 /home/yuya/ds41/refusal_check.py --url http://127.0.0.1:8150 --reasoning-effort none > "logs/qtest-$NAME-refusal.txt" 2>&1
  grep -cE "^answered" "logs/qtest-$NAME-refusal.txt" | sed 's/^/   answered: /'
fi
trap - EXIT
restore
if ! grep -q "\"label\": \"$PROD\"" /home/yuya/ds41/results.jsonl 2>/dev/null; then
  echo "$(date +%T) production speed row"
  (cd /home/yuya/ds41 && python3 bench.py --url http://127.0.0.1:8150 --model deepseek-v4.1-flash-1017 --label "$PROD" --reasoning-effort none 2>&1 | grep -E "^$PROD|rror" | cut -c1-200)
fi
if [ ! -s "logs/qtest-$NAME-refusal-prod.txt" ]; then
  echo "$(date +%T) refusal check on production (thinking off)"
  python3 /home/yuya/ds41/refusal_check.py --url http://127.0.0.1:8150 --reasoning-effort none > "logs/qtest-$NAME-refusal-prod.txt" 2>&1
fi
{ echo "chain-qtest $NAME report $(date '+%F %T')"
  echo "held-out 8x1024 (ppl / top-1 / KL top-256 vs the FP8 teacher):"
  echo "  $TAG  $(cat "local/eval-lucebox/$TAG/score.json" 2>/dev/null)"
  echo "  base-3gpu  $(cat local/eval-lucebox/base-3gpu/score.json)"
  echo "  mayas-abl  $(cat local/eval-lucebox/mayas-abl/score.json 2>/dev/null)"
  echo "security set (KL(prod || candidate), top-1 vs production, both ppl):"
  echo "  $TAG  $(cat "local/secprof/eval/$TAG/score.json" 2>/dev/null)"
  echo "  mayas-abl  $(cat local/secprof/eval/mayas-abl/score.json 2>/dev/null)"
  echo "speed (bench.py rows, thinking off):"
  (cd /home/yuya/ds41 && python3 table.py "$PROD" "$LABEL" M-prod M-mayas 2>&1)
  echo "refusal check (thinking off): candidate $(grep -c '^answered' "logs/qtest-$NAME-refusal.txt" 2>/dev/null)/4 answered, production $(grep -c '^answered' "logs/qtest-$NAME-refusal-prod.txt" 2>/dev/null)/4 answered"
} > "$REP"
cat "$REP"
echo "== chain-qtest $NAME done $(date '+%F %T')"
