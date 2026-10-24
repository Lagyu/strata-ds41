#!/bin/bash
# chain-orca.sh PART - the Mix-256G comparison (2026-10-25; user 2026-10-24: compare Mix-256G with our MXFP4+ABL AND with the
# original without ABL, without cold-cache bias in the tokens/s). Two windows, so that production is back in between:
#   PART 1 (quality, ~75 min): waits for the preparation (local/qtest/orca-moe-test.ok) and an IDLE server, stops production,
#     switches the internal-SSD twin of production's file to the ORIGINAL dense bytes (= "MXFP4 without ABL"), held-out text vs the FP8
#     original (Mix-256G exact, original-no-ABL, Mix-256G with --skip-file 0.10), security code (the original-no-ABL run becomes a
#     reference; Mix-256G and production are scored against it and against production's old reference = comparable with Maya-S /
#     GSQ-RCO), the refusal check on the original, twin back, production back.  Marker: local/qtest/orca-part1.end
#   PART 2 (speed, ~100 min): waits for part 1's marker and an IDLE server, then per configuration: [cold pass right after the start]
#     -> warm-up (12 varied prompts x 256 tokens) -> measured passes. Production as deployed (--skip-file 0.10) and exact, Mix-256G
#     the same two ways, production-as-deployed again (drift check). Production back, report logs/orca-report.txt.
# "Idle" = engine log untouched for IDLE_MIN minutes and GPUs <= 10 %; while a window lasts the log is kept fresh (other sessions'
# chains wait on it). Production is restored by an EXIT trap whatever happens (only if it was running before). Each step skips when
# its result exists. Launch (survive harness restarts):
#   sudo systemd-run --unit=orca-chain1 --property=User=yuya --property=LimitMEMLOCK=infinity --property=LimitNOFILE=65536 --collect \
#        /bin/bash /home/yuya/llm-src/strata-ds41/local/qtest/chain-orca.sh 1
#   sudo systemd-run --unit=orca-chain2 ... /bin/bash /home/yuya/llm-src/strata-ds41/local/qtest/chain-orca.sh 2
set -u
PART=${1:-1}
R=/home/yuya/llm-src/strata-ds41; cd "$R" || exit 1
Q=local/qtest; O=/home/yuya/ds41/orca
BIN=$R/build-hip-mayas/ds41_generate
PY=/home/yuya/llm-src/hfenv/bin/python
PROD=/mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf
TWIN=/home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf
MIX=/mnt/win/llm/orca/DS41-1017-Mix256G-00001-of-00003.gguf
IDLE_MIN=${IDLE_MIN:-15}
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/yuya
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p logs
exec >> logs/orca-window.log 2>&1
echo "== chain-orca part $PART start $(date '+%F %T')"
LAN=/home/yuya/.config/strata-ds41/lan.env
[ -f "$LAN" ] && { set -a; . "$LAN"; set +a; }
. "$Q/orca-lib.sh"
if [ "$PART" = 1 ]; then
  until [ -e "$Q/orca-moe-test.ok" ]; do
    [ -e "$Q/orca-moe-test.FAILED" ] && { echo "$(date +%T) preparation FAILED ($(head -1 "$Q/orca-moe-test.FAILED")) - no window"; echo "failed: preparation" > "$Q/orca-part1.end"; exit 1; }
    sleep 60
  done
  echo "$(date +%T) preparation OK: $(head -1 "$Q/orca-moe-test.ok")"
else
  until [ -e "$Q/orca-part1.end" ]; do sleep 60; done
  echo "$(date +%T) part 1 ended: $(head -1 "$Q/orca-part1.end")"
fi
[ -x "$BIN" ] && [ -s "$MIX" ] || { echo "missing $BIN or $MIX"; exit 1; }
EL=local/serve-engine.log; note=0
while true; do
  now=$(date +%s); age=$(( (now - $(stat -c %Y "$EL")) / 60 ))
  busy=$(rocm-smi --showuse 2>/dev/null | grep "GPU use" | grep -oE "[0-9]+$" | sort -n | tail -1)
  if [ "$age" -ge "$IDLE_MIN" ] && [ "${busy:-100}" -le 10 ]; then
    echo "$(date +%T) idle: the engine log untouched for $age min, max GPU use ${busy}%"; break
  fi
  if [ $((now - note)) -ge 1800 ]; then echo "$(date +%T) waiting for an idle server (engine log age $age min, max GPU use ${busy:-?}%)"; note=$now; fi
  sleep 60
done
WIN=$Q/.orca-window; touch "$WIN"   # the window marker other sessions' chains know: keep the engine log fresh while it lasts
( while [ -e "$WIN" ]; do touch "$EL"; sleep 10; done ) &
GUARD=$!
WAS=$(systemctl is-active strata-ds41.service)
finish() {   # restore, and make sure part 2 is not left waiting for a part 1 that died
  restore
  if [ "$PART" = 1 ] && [ ! -e "$Q/orca-part1.end" ]; then echo "ended early (see logs/orca-window.log) $(date '+%F %T')" > "$Q/orca-part1.end"; fi
}
trap finish EXIT
echo "$(date +%T) stopping strata-ds41 (was $WAS)"
sudo systemctl stop strata-ds41.service; wait_vram
if [ "$PART" = 1 ]; then
  "$PY" -I "$O/twin_noabl.py" to-original || { echo "twin switch failed"; echo "failed: twin switch" > "$Q/orca-part1.end"; exit 1; }
  "$PY" -I "$O/twin_noabl.py" status
  # ---- preflight with production stopped: the tests that hung / ran out of memory beside it
  mmq_preflight
  cpu_preflight
  MA=$(mix_args)
  # ---- held-out text vs the FP8 original (exact routing), then Mix-256G with the skip rule production now runs
  heldout orca-mix256 "$MIX" $MA
  heldout stock-mxfp4 "$TWIN"
  # ---- security code: the original-no-ABL run is the reference; the production run doubles as the binary parity check
  if [ ! -s local/secprof/eval/teacher-stock.npz ]; then
    echo "$(date +%T) security reference from the original-no-ABL model"
    BIN="$BIN" bash "$Q/run-eval-sec-multi.sh" stock-ref "$TWIN" teacher local/secprof/eval/teacher-stock.npz 2>&1 | tail -6
  fi
  ENGINE_ARGS="$MA" secure orca-sec "$MIX" local/secprof/eval/teacher-stock.npz local/secprof/eval/teacher-prod.npz
  secure prod-sec "$PROD" local/secprof/eval/teacher-stock.npz local/secprof/eval/teacher-prod.npz
  heldout orca-mix256-skip10 "$MIX" $MA --skip-file 0.10
  # ---- refusal check on the original, then the twin back
  if [ ! -s logs/orca-refusal-stock.txt ]; then
    serve_start "$Q/ds41-yuya-3gpu-stock.json" && refusal_to logs/orca-refusal-stock.txt
    sudo systemctl stop strata-ds41.service; wait_vram
  fi
  twin_back
  trap - EXIT
  restore
  echo "done $(date '+%F %T')" > "$Q/orca-part1.end"
  "$PY" -I "$O/report_orca.py" > logs/orca-report-part1.txt 2>&1
  echo "== chain-orca part 1 done $(date '+%F %T')"
else
  # ---- the refusal check on the original (part 1's attempt died: the old test binary did not know the server's --batch-slots)
  if [ ! -s logs/orca-refusal-stock.txt ]; then
    "$PY" -I "$O/twin_noabl.py" to-original && "$PY" -I "$O/twin_noabl.py" status
    serve_start "$Q/ds41-yuya-3gpu-stock.json" && refusal_to logs/orca-refusal-stock.txt
    sudo systemctl stop strata-ds41.service; wait_vram
    twin_back
  fi
  # ---- speed, warm protocol: production as deployed (skip rule) vs Mix-256G the same way, then both exact, then a drift check
  MIXSKIP=$(mix_cfg "$Q/ds41-yuya-3gpu-mix256-skip.json"); MIXEXACT=$(mix_cfg "$Q/ds41-yuya-3gpu-mix256.json")
  echo "Mix configs: $MIXSKIP, $MIXEXACT"
  speed_cfg prodskip1 "$Q/ds41-yuya-3gpu-prodbench-skip.json" 1 3
  speed_cfg mixskip "$MIXSKIP" 1 3; [ $? -eq 0 ] && refusal_to logs/orca-refusal-mix.txt
  speed_cfg prodexact "$Q/ds41-yuya-3gpu-prodbench.json" 0 2; [ $? -eq 0 ] && refusal_to logs/orca-refusal-prod.txt
  speed_cfg mixexact "$MIXEXACT" 0 2
  speed_cfg prodskip2 "$Q/ds41-yuya-3gpu-prodbench-skip.json" 0 2
  trap - EXIT
  restore
  "$PY" -I "$O/report_orca.py" > logs/orca-report.txt 2>&1
  cat logs/orca-report.txt
  echo "== chain-orca part 2 done $(date '+%F %T')"
fi
