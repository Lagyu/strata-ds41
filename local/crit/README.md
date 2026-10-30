# local/crit - the prefill + decode critical-path work (2026-10-30/31)

`report.md` is the write-up: where the time goes (measured), every optimization tried with numbers, what was deployed
(2026-10-31 02:13: the RDNA matrix-core lightning indexer `STRATA_INDEXER_WMMA=1`, the queued chunk sum
`STRATA_CHUNK_ASYNC=1`, the fused hc kernels and the skip-read exclusion), what was rejected and why, and two hardware
findings (the board's PCIe 4.0 root port is the DMA ceiling; the R9700 is not stable undervolted under a 98K read).

Tooling (paths as used on this machine; the work dir was /home/yuya/ds41/crit, logs and arm outputs are not in the repo):

- `window1.sh` .. `window5.sh` - the measurement windows: wait for other GPU units, wait for the server to go quiet, stop
  production, run the arms (the std probe `local/perf-probe/run.py`, the long probe `run-long.py`, one-shot smokes, the
  held-out eval `ho-eval.sh`), restart production from an EXIT trap.  Run as transient systemd units.
- `run-long.py` / `run-long-short.py` - the 98K-context probe (a fresh 98K read, appends of 150..2500 tokens, a bench).
- `oneshot.sh` + `oneshot-env.sh` - a one-shot prefill with production's args and env (exactness smokes).
- `build*.sh`, `ggml-obj.sh` - how the candidate binaries were built: ggml objects compiled with the main build's
  compile command, swapped into a copy of the prebuilt ggml archive, the engine relinked against it.
- `rocprof-engine.sh`, `rocprof_summary.py` - a kernel trace of the engine and its per-kernel summary.
- `cmp_lidx.py` - compares two lightning-indexer score dumps (`test_ds41_indexer LIDX_DUMP=...`).
- `deploy-crit.sh` - the deploy: config env additions, binary install beside the running one, restart, smoke.
- `dmabench/dmabench.cpp` - pinned host -> device copy bandwidth on 1 / 2 / 4 streams (`hipcc -O2 -o dmabench dmabench.cpp`).

Payloads are public text (Project Gutenberg token ids built by `local/perf-probe/build_payload.py`; the security payload
from the NIST Juliet set); they are `.i32` files and stay out of the repo.
