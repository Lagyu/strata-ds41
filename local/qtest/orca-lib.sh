# orca-lib.sh - helpers for chain-orca.sh (sourced). Expects R, Q, O, BIN, PY, TWIN already set.
wait_vram() {
  local i b
  for i in $(seq 1 90); do
    b=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk -F': ' '{s+=$NF} END{print s+0}')
    [ "$b" = "0" ] && break
    sleep 2
  done
  echo "$(date +%T) VRAM released after ~$((i * 2)) s"
}
twin_back() {   # the internal-SSD twin back to the abliterated (production) bytes, verified against the recorded hashes
  "$PY" -I "$O/twin_noabl.py" to-abl || echo "TWIN RESTORE FAILED: run  $PY -I $O/twin_noabl.py to-abl  by hand"
}
restore() {
  echo "$(date +%T) cleanup: stopping the service, twin back to the abliterated bytes, production back"
  sudo systemctl stop strata-ds41.service
  wait_vram
  twin_back
  echo local/ds41-yuya-3gpu.json > local/active-config
  if [ "${WAS:-}" = active ]; then bash /home/yuya/dsh-work/tools/restart-ds41.sh
  else echo "production was not running before the window: left stopped"; fi
  rm -f "${WIN:-/nonexistent-window-marker}"; [ -n "${GUARD:-}" ] && kill "$GUARD" 2>/dev/null
}
serve_start() {   # CONFIG.json - stop, switch the config, start, wait until the API answers (systemctl start returns after warmup.sh)
  local code i
  sudo systemctl stop strata-ds41.service
  wait_vram
  echo "$1" > local/active-config
  sudo systemctl start strata-ds41.service || { echo "start failed for $1"; return 1; }
  for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w '%{http_code}' ${STRATA_API_KEY:+-H "Authorization: Bearer $STRATA_API_KEY"} http://127.0.0.1:8150/v1/models 2>/dev/null)
    [ "$code" = "200" ] && break
    systemctl is-active -q strata-ds41.service || { echo "the service died while loading $1"; return 1; }
    sleep 5
  done
  [ "$code" = "200" ] && echo "$(date +%T) service ready on $1"
}
bench_run() {   # LABEL [bench.py args]
  local lab=$1; shift
  (cd /home/yuya/ds41 && python3 bench.py --url http://127.0.0.1:8150 --model deepseek-v4.1-flash-1017 --label "$lab" --reasoning-effort none "$@" 2>&1 \
     | grep -E "^$lab|rror" | cut -c1-180)
}
warm_up() {   # 12 varied prompts x 256 tokens: fills the page cache, the arena, the VRAM LRU, the graphs and the drafter's state
  (cd /home/yuya/ds41 && python3 bench.py --url http://127.0.0.1:8150 --model deepseek-v4.1-flash-1017 --calibrate --calib-set 1 --calib-tokens 256 \
     --reasoning-effort none 2>&1 | tail -3 | cut -c1-120)
}
speed_cfg() {   # NAME CONFIG COLD(0|1) REPS : [cold pass] -> warm-up -> measured passes
  local name=$1 cfg=$2 cold=$3 reps=$4
  grep -q "\"label\": \"S-orca-$name\"" /home/yuya/ds41/results.jsonl 2>/dev/null && { echo "speed $name: done earlier"; return 2; }
  serve_start "$cfg" || return 1
  if [ "$cold" = 1 ]; then echo "$(date +%T) $name: cold pass (first bench set after the start)"; bench_run "C-orca-$name"; fi
  echo "$(date +%T) $name: warm-up"; warm_up
  echo "$(date +%T) $name: $reps measured passes"; bench_run "S-orca-$name" --repeat "$reps"
}
refusal_to() {   # OUTFILE  (the service is up on the config to test)
  [ -s "$1" ] && { echo "refusal check: $1 exists"; return 0; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' ${STRATA_API_KEY:+-H "Authorization: Bearer $STRATA_API_KEY"} http://127.0.0.1:8150/v1/models 2>/dev/null)" = 200 ] \
    || { echo "refusal check: the API is not answering"; return 1; }
  python3 /home/yuya/ds41/refusal_check.py --url http://127.0.0.1:8150 --reasoning-effort none > "$1" 2>&1
  echo "$(date +%T) refusal check -> $1: $(grep -c '^answered' "$1") answered, $(grep -c '^REFUSED' "$1") refused"
}
heldout() {   # TAG MODEL [extra engine args, e.g. --skip-file 0.10]
  local tag=$1 model=$2; shift 2
  if [ ! -s "local/eval-lucebox/$tag/score.json" ]; then
    echo "$(date +%T) held-out eval $tag $*"
    if [ $# -gt 0 ]; then BIN="$BIN" MODEL="$model" bash local/maya-s/run-eval-mayas.sh "$tag" -- "$@" 2>&1 | tail -9
    else BIN="$BIN" MODEL="$model" bash local/maya-s/run-eval-mayas.sh "$tag" 2>&1 | tail -9; fi
  fi
  echo "held-out $tag: $(cat "local/eval-lucebox/$tag/score.json" 2>/dev/null)"
}
secure() {   # TAG MODEL REFS...
  local tag=$1 model=$2; shift 2
  if [ ! -s "local/secprof/eval/$tag/score-teacher-stock.json" ]; then
    echo "$(date +%T) security eval $tag"
    BIN="$BIN" bash "$Q/run-eval-sec-multi.sh" "$tag" "$model" score "$@" 2>&1 | tail -6
  fi
  echo "security $tag: $(cat "local/secprof/eval/$tag/score-teacher-stock.json" 2>/dev/null)"
}

# ---- preflight (inside the quality window, production stopped): the tests that hung / ran out of memory beside production
FLAGS=$Q/orca-mix-flags.txt
mmq_preflight() {   # every expert class's prompt-path MMQ chunk on the XTX (gfx1100, like card 0), a hard timeout per layer
  local L rc bad="" good=""
  if [ -s "$FLAGS" ]; then echo "preflight: decided earlier: $(cat "$FLAGS")"; return 0; fi
  : > logs/orca-preflight.txt
  for L in 0 4 5 13 15 23 24 34; do
    timeout -k 10 150 env HIP_VISIBLE_DEVICES=2 stdbuf -oL "$R/build-hip-mayas/test_ds41_moe_gpu" "$MIX" --layers "$L" --tokens 1 --gpu --chunk 32 --chunk-mmq \
      > "/tmp/orca-pre-$L.out" 2>&1
    rc=$?
    echo "layer $L: exit $rc  $(grep -E '^40 layers|chunk 32 tokens|^stats' "/tmp/orca-pre-$L.out" | tr '\n' ' ' | cut -c1-200)" >> logs/orca-preflight.txt
    if [ $rc -eq 0 ]; then good="$good $L"; else bad="$bad $L"; wait_vram; fi
  done
  echo "preflight MMQ on gfx1100: pass:${good:- none}; fail/hang:${bad:- none}"
  if [ -n "$bad" ]; then echo "--no-chunk-mmq" > "$FLAGS"; else echo "none" > "$FLAGS"; fi
}
cpu_preflight() {   # the CPU tier's expert kernels on the same classes (no GPU), also bounded
  [ -s logs/orca-cpu-preflight.txt ] && { echo "cpu preflight: done earlier: $(tail -1 logs/orca-cpu-preflight.txt | cut -c1-100)"; return 0; }
  timeout -k 10 900 "$R/build-hip-mayas/test_ds41_moe" "$MIX" --layers 0,4,5,13,15,23,24,34 --tokens 4 > logs/orca-cpu-preflight.txt 2>&1
  echo "cpu preflight exit $?: $(grep '^stats' logs/orca-cpu-preflight.txt | cut -c1-120)"
}
mix_args() {   # the extra engine args every Mix run needs (from the preflight)
  [ "$(cat "$FLAGS" 2>/dev/null)" = "--no-chunk-mmq" ] && echo "--no-chunk-mmq"
}
mix_cfg() {   # SRC.json -> prints the config to use (with --no-chunk-mmq added when the preflight asked for it)
  local a; a=$(mix_args)
  if [ -z "$a" ]; then echo "$1"; return; fi
  python3 -I - "$1" "${1%.json}-nommq.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["args"] += ["--no-chunk-mmq"]
c["_comment"] += " | VARIANT: --no-chunk-mmq (the preflight found a hanging prompt-path kernel on gfx1100)"
json.dump(c, open(sys.argv[2], "w"), indent=1); print(sys.argv[2])
PY
}
