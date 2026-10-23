#!/bin/bash
# This box: the server with the config named in local/active-config (default local/ds41-yuya-xtx.json).
# API on 127.0.0.1:8150 (OpenAI + Anthropic), web chat at /. MCP servers for the web chat's tools (DuckDuckGo search):
# local/mcp.json, passed as --mcp-config when it exists, whichever config is active.
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R" || exit 1
export STRATA_HOME="$R"
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CFG=$(cat local/active-config 2>/dev/null || echo local/ds41-yuya-xtx.json)
VENV=${STRATA_VENV:-/home/yuya/llm-src/Strata-DSV4.1/.venv}
MCP=()
[ -f local/mcp.json ] && MCP=(--mcp-config "$R/local/mcp.json")
# LAN access (the web chat from a phone): ~/.config/strata-ds41/lan.env sets DS41_HOST=0.0.0.0 and STRATA_API_KEY (the
# server reads the key from the environment and then wants it from every client, local scripts too). No file = 127.0.0.1.
LAN=/home/yuya/.config/strata-ds41/lan.env
[ -f "$LAN" ] && { set -a; . "$LAN"; set +a; }
HOST=()
[ -n "${DS41_HOST:-}" ] && HOST=(--host "$DS41_HOST")
# STRATA_ROUTE_COUNTS (expert-use counts of the real traffic): the engine starts its counts from zero and overwrites the
# file, so keep the previous run's file under its end time (sum them offline: counts-*.csv + counts.csv)
RC=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('env', {}).get('STRATA_ROUTE_COUNTS', ''))" "$R/$CFG" 2>/dev/null)
[ -n "$RC" ] && [ -s "$RC" ] && mv "$RC" "${RC%.csv}-$(date -r "$RC" +%Y%m%d-%H%M%S).csv"
echo "run-ds41: config $CFG${MCP:+, MCP servers from local/mcp.json}${HOST:+, listening on $DS41_HOST (API key from $LAN)}"
exec "$VENV/bin/python" "$R/tools/v41/serve/serve_ds41.py" --engine strata --config "$R/$CFG" --port 8150 "${HOST[@]}" "${MCP[@]}" "$@"
