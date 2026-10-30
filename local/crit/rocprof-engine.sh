#!/bin/bash
# rocprof-engine.sh - the engine under rocprofv3 (kernel trace + stats): DS41_EXE for one probe run.  ROCPROF_DIR = output dir.
exec rocprofv3 --kernel-trace --stats -d "${ROCPROF_DIR:-/tmp/rocprof}" -o prof --output-format csv -- \
  /home/yuya/llm-src/strata-zcnvis/build-hip-deploy/ds41_generate "$@"
