#!/bin/bash
# The engine binary the configs name as "exe": build-hip/ds41_generate with ROCm's libraries on the path.
# DS41_EXE overrides the binary.
R=$(cd "$(dirname "$0")/.." && pwd)
export LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec "${DS41_EXE:-$R/build-hip/ds41_generate}" "$@"
