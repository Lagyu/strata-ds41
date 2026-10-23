#!/bin/bash
# local/sweep.sh LABEL [BASE.json] [KEY=VALUE ...] - one benchmark run of a configuration variant: the base config
# (default: local/active-config's) with env overrides (KEY=VALUE -> "env") and argument overrides (--flag=VALUE ->
# "args"; --flag= removes the flag), written to local/variants/LABEL.json; the service restarted on it, the model
# reloaded, then /home/yuya/ds41/bench.py's prompt set (thinking off, as the Lucebox reference rows) under LABEL.
# Lines of the engine's own per-request reports are echoed at the end.  Runs ~8-12 minutes; chain several in a script.
#   local/sweep.sh pcie07 local/ds41-yuya-3gpu.json STRATA_TIER_BATCH_PCIE=0.7
#   local/sweep.sh thr20 "" --threads=20
set -u
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R" || exit 1
LABEL=$1; shift
BASE=${1:-}; [ $# -gt 0 ] && shift
[ -z "$BASE" ] && BASE=$(cat local/active-config 2>/dev/null || echo local/ds41-yuya-3gpu.json)
mkdir -p local/variants
OUT=local/variants/$LABEL.json
python3 - "$BASE" "$OUT" "$@" <<'EOF'
import json, sys
base, out, *kv = sys.argv[1:]
cfg = json.load(open(base))
args = cfg["args"]
for item in kv:
    k, _, v = item.partition("=")
    if k.startswith("--"):
        if k in args:
            i = args.index(k)
            # a flag with a value, or a bare flag
            has_val = i + 1 < len(args) and not str(args[i + 1]).startswith("--")
            del args[i:i + (2 if has_val else 1)]
        if v != "":
            args += [k, v] if v != "on" else [k]
    else:
        if v == "":
            cfg["env"].pop(k, None)
        else:
            cfg["env"][k] = v
cfg["_comment"] = f"sweep variant of {base}: {' '.join(kv)}"
json.dump(cfg, open(out, "w"), indent=1)
print("variant:", out, "<-", base, *kv)
EOF
echo "$OUT" > local/active-config
# (2026-10-23: the readiness check sends STRATA_API_KEY when the environment has it - the server wants it since lan.env)
# stop, wait until every card's VRAM is released (a start right after a stop found 8 GiB still held on the primary:
# the dense weights' cudaMalloc failed), then start
sudo systemctl stop strata-ds41.service
for i in $(seq 1 30); do
  busy=$(rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" | awk -F': ' '{s+=$NF} END{print s+0}')
  [ "$busy" = "0" ] && break
  sleep 2
done
sudo systemctl start strata-ds41.service || exit 1
sleep 5
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' ${STRATA_API_KEY:+-H "Authorization: Bearer $STRATA_API_KEY"} http://127.0.0.1:8150/v1/models 2>/dev/null)
  [ "$code" = "200" ] && break
  systemctl is-active -q strata-ds41.service || { echo "$LABEL: the service died while loading"; tail -5 local/serve-engine.log | cut -c1-200; exit 1; }
  sleep 5
done
[ "$code" = "200" ] || { echo "$LABEL: not ready after $((i*5)) s"; exit 1; }
echo "$LABEL: ready after $((i*5)) s ($(date +%T))"
START=$(wc -l < local/serve-engine.log)
(cd /home/yuya/ds41 && python3 bench.py --url http://127.0.0.1:8150 --model deepseek-v4.1-flash-1017 --label "$LABEL" --reasoning-effort none ${BENCH_ARGS:-} 2>&1 | grep -E "^$LABEL|Error|error" | cut -c1-200)
tail -n +"$START" local/serve-engine.log | grep -E "^serve: (prompt [0-9]+ tokens|[0-9]+ tokens in|DSpark [0-9]+ passes)|^peer:|^tier:" | cut -c1-230
echo "$LABEL: done $(date +%T)"
