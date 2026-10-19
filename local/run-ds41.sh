#!/bin/bash
# This box: the server with the config named in local/active-config (default local/ds41-yuya-xtx.json).
# API on 127.0.0.1:8150 (OpenAI + Anthropic), web chat at /.
R=$(cd "$(dirname "$0")/.." && pwd); cd "$R" || exit 1
export STRATA_HOME="$R"
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CFG=$(cat local/active-config 2>/dev/null || echo local/ds41-yuya-xtx.json)
VENV=${STRATA_VENV:-/home/yuya/llm-src/Strata-DSV4.1/.venv}
echo "run-ds41: config $CFG"
exec "$VENV/bin/python" "$R/tools/v41/serve/serve_ds41.py" --engine strata --config "$R/$CFG" --port 8150 "$@"
