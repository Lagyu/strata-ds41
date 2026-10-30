#!/bin/bash
# oneshot.sh TAG BIN IDS [ENV=VAL ...] - one-shot prefill (-n 1) of IDS with production's args/env (+ the given), log to oneshot-TAG.log
set -u
TAG=$1 BIN=$2 IDS=$3; shift 3
. /home/yuya/ds41/crit/oneshot-env.sh
cd /home/yuya/llm-src/strata-ds41 || exit 1
export LD_LIBRARY_PATH=/opt/rocm/lib
eval "env $ENV $* timeout -k 20 900 $BIN $ARGS --ids-file $IDS -n 1 --temp 0" > /home/yuya/ds41/crit/oneshot-$TAG.out 2> /home/yuya/ds41/crit/oneshot-$TAG.log
echo "exit $?"
