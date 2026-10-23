#!/bin/bash
# chain-mayas.sh - the Maya-S measurement window. Production is STOPPED for ~1.5-2 h and restored by the EXIT trap (on the
# production config local/ds41-yuya-3gpu.json, via ~/dsh-work/tools/restart-ds41.sh) whatever happens.
# Prepared 2026-10-23 without downtime: /mnt/ext/models/maya-s-1017/MERGE.done (merge.sh: Maya-S routed experts + the
# production file's abliterated dense tensors in one Lucebox-style GGUF), build-hip-mayas/{ds41_generate,
# native_grouped_parity} (build.sh "BUILD OK": the IQ3_S down kernel), local/maya-s/ds41-yuya-3gpu-mayas.json.
# Launch:
#   sudo systemd-run --unit=mayas-chain --property=User=yuya --property=LimitMEMLOCK=infinity --property=LimitNOFILE=65536 \
#        --collect /bin/bash /home/yuya/llm-src/strata-ds41/local/maya-s/chain-mayas.sh
# Steps (each skipped when its result exists, so a rerun continues): 1 copy the merged file to the P41 (~8 min);
# 2 kernel parity test; 3 held-out eval (local/eval-lucebox/mayas-abl/score.json vs base-3gpu: 8.626 / 93.56% / 0.0448);
# 4 security set (production's reference first if missing, then local/secprof/eval/mayas-abl/score.json); 5 the speed set
# on the Maya-S config (sweep M-mayas); production back; a same-day production row (M-prod).
# Report: logs/chain-mayas-report.txt. Log: logs/chain-mayas.log.
set -u
R=/home/yuya/llm-src/strata-ds41; M=$R/local/maya-s
SRC=/mnt/ext/models/maya-s-1017/DS41-1017-MayaS-ABL.gguf
DST=/mnt/win/llm/maya-s/DS41-1017-MayaS-ABL.gguf
BIN=$R/build-hip-mayas/ds41_generate; PAR=$R/build-hip-mayas/native_grouped_parity
export PATH=/opt/rocm/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/yuya
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$R" || exit 1
mkdir -p logs
exec >> logs/chain-mayas.log 2>&1
echo "== chain-mayas start $(date '+%F %T')"
[ -e /mnt/ext/models/maya-s-1017/MERGE.done ] || { echo "MERGE.done missing (merge.sh not finished)"; exit 1; }
[ -x "$BIN" ] && [ -x "$PAR" ] || { echo "build-hip-mayas binaries missing (build.sh)"; exit 1; }
grep -q "BUILD OK" /home/yuya/ds41/maya-s/build.log || { echo "build.log has no BUILD OK"; exit 1; }
LAN=/home/yuya/.config/strata-ds41/lan.env
[ -f "$LAN" ] && { set -a; . "$LAN"; set +a; }
wait_vram() {
  for i in $(seq 1 90); do
    busy=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk -F': ' '{s+=$NF} END{print s+0}')
    [ "$busy" = "0" ] && break
    sleep 2
  done
  echo "$(date +%T) VRAM released after ~$((i * 2)) s"
}
restore() {
  echo "$(date +%T) restoring production (local/ds41-yuya-3gpu.json)"
  echo local/ds41-yuya-3gpu.json > local/active-config
  bash /home/yuya/dsh-work/tools/restart-ds41.sh
}
trap restore EXIT
echo "$(date +%T) stopping strata-ds41"
sudo systemctl stop strata-ds41.service; wait_vram
# 1) the merged file on the P41: every engine start reads the whole model (all experts resident); from the USB SSD that
#    would cost ~6 min per start, and the speed rows want the file tier on NVMe as production has it
if [ ! -e "$DST" ] || [ "$(stat -c %s "$DST")" != "$(stat -c %s "$SRC")" ]; then
  echo "$(date +%T) copying $(stat -c %s "$SRC") bytes to $DST"
  mkdir -p "$(dirname "$DST")"; rm -f "$DST"
  cp "$SRC" "$DST" || { echo "copy failed"; exit 1; }
  echo "$(date +%T) copy done"
fi
# 2) the kernel parity test (the IQ3_S down path and Maya-S's other pairs, bitwise against the v1 launches)
if [ ! -e "$M/parity.ok" ]; then
  echo "$(date +%T) native_grouped_parity on card 0"
  HIP_VISIBLE_DEVICES=0 "$PAR" > logs/mayas-parity.log 2>&1
  tail -3 logs/mayas-parity.log
  if grep -q "native_grouped_parity: 0 failures" logs/mayas-parity.log; then cp logs/mayas-parity.log "$M/parity.ok"; else echo "PARITY FAILED - stopping"; exit 1; fi
fi
# 3) the held-out eval (Lucebox's 8 x 1024 vs the FP8 teacher), the residency and scorer of base-3gpu
if [ ! -s local/eval-lucebox/mayas-abl/score.json ]; then
  echo "$(date +%T) held-out eval"
  BIN="$BIN" MODEL="$DST" bash "$M/run-eval-mayas.sh" mayas-abl 2>&1 | tail -9
fi
echo "mayas-abl: $(cat local/eval-lucebox/mayas-abl/score.json 2>/dev/null)"
echo "base-3gpu: $(cat local/eval-lucebox/base-3gpu/score.json)"
# 4) the security held-out set: production's reference (never made yet), then Maya-S against it
if [ ! -s local/secprof/eval/teacher-prod.npz ]; then
  echo "$(date +%T) security reference from production (run-eval-sec.sh prod)"
  bash local/secprof/run-eval-sec.sh prod 2>&1 | tail -5
fi
if [ ! -s local/secprof/eval/mayas-abl/score.json ]; then
  echo "$(date +%T) security eval"
  BIN="$BIN" bash "$M/run-eval-sec-mayas.sh" mayas-abl "$DST" 2>&1 | tail -5
fi
echo "security mayas-abl: $(cat local/secprof/eval/mayas-abl/score.json 2>/dev/null)"
# 5) speed: the service on the Maya-S config + bench.py's set (sweep.sh leaves the service on it; restore() puts production back)
if ! grep -q '"label": "M-mayas"' /home/yuya/ds41/results.jsonl 2>/dev/null; then
  echo "$(date +%T) speed set on the Maya-S config"
  bash local/sweep.sh M-mayas "$M/ds41-yuya-3gpu-mayas.json" 2>&1 | tail -12
fi
trap - EXIT
restore
# a same-day production row for the table (production is serving again; this competes only mildly with real requests)
if ! grep -q '"label": "M-prod"' /home/yuya/ds41/results.jsonl 2>/dev/null; then
  echo "$(date +%T) production speed row"
  (cd /home/yuya/ds41 && python3 bench.py --url http://127.0.0.1:8150 --model deepseek-v4.1-flash-1017 --label M-prod --reasoning-effort none 2>&1 | grep -E "^M-prod|rror" | cut -c1-200)
fi
{ echo "chain-mayas report $(date '+%F %T')"
  echo "held-out 8x1024 (ppl / top-1 / KL top-256 vs the FP8 teacher):"
  echo "  mayas-abl  $(cat local/eval-lucebox/mayas-abl/score.json 2>/dev/null)"
  echo "  base-3gpu  $(cat local/eval-lucebox/base-3gpu/score.json)"
  echo "security set (KL(prod || mayas-abl), top-1 vs production, both ppl):"
  echo "  $(cat local/secprof/eval/mayas-abl/score.json 2>/dev/null)"
  echo "speed (bench.py rows, thinking off):"
  (cd /home/yuya/ds41 && python3 table.py M-prod M-mayas 2>&1)
} > logs/chain-mayas-report.txt
cat logs/chain-mayas-report.txt
echo "== chain-mayas done $(date '+%F %T')"
