# strata-ds41: DeepSeek-V4.1-Flash on one AMD workstation

This is the LLM server of one home workstation: the DeepSeek-V4.1 engine of
[Xolotl](https://github.com/Indras-Mirror/Xolotl) (formerly Strata-DSV4.1) with the AMD work of vincalkr's fork
([vincalkr/DS41Engine](https://github.com/vincalkr/DS41Engine)), merged onto [Strata](https://github.com/Niko1221/Strata)
0.2.0, plus what was done on this machine: the ROCm build on three GPUs, the serving work below and the machine's
configuration. The upstream history is kept as it is; the work on top of it is in the commits by `yuya` (2026-10-20
onwards). MIT licensed, like upstream (`LICENSE`).

## The machine

- Intel Core i7-14700K, 128 GB DDR4; Ubuntu 26.04 (kernel 7.0), ROCm 7.14
- GPU 0: Radeon RX 7900 XTX 24 GB on PCIe x16, the primary (dense layers, attention, the most used experts)
- GPU 1: Radeon AI PRO R9700 32 GB, an OCuLink eGPU on PCIe 4.0 x4: a peer expert store and the DSpark drafter
- GPU 2: Radeon RX 7900 XTX 24 GB on a chipset PCIe x4 slot: a peer expert store
- Model: DeepSeek-V4.1-Flash-1017 with MXFP4 experts (an abliterated build; 15,360 experts, ~269 GiB). The experts are
  tiered over the primary's VRAM, a 95 GiB pinned host arena, the two peer stores and the model file on NVMe.

## Build (HIP, gfx1100 + gfx1201)

```sh
bash tools/xolotl/fetch-llama.sh    # third_party/llama.cpp at the pinned commit + tools/xolotl/ggml/xolotl-ggml.patch
tools/v41/build-ggml-hip.sh         # its ggml with the HIP backend, static (the flash-attention instances take long)
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ -DCMAKE_HIP_ARCHITECTURES="gfx1100;gfx1201" \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp -DSTRATA_GGML_PREBUILT=$PWD/third_party/llama.cpp/build-hip
cmake --build build-hip --target ds41_generate
```

The server is Python (`requirements.txt`); `local/run-ds41.sh` takes its virtualenv from `STRATA_VENV`.

## Running it

- `local/ds41-yuya-3gpu.json`: the production config. Its `_comment` records each setting and the measurement behind
  it; `local/variants/` holds the sweep rows those measurements came from.
- `local/run-ds41.sh`: the launcher (`tools/v41/serve/serve_ds41.py`: an OpenAI- and Anthropic-compatible API and a
  web chat on port 8150); `local/warmup.sh` sends the first request after a start.
- `local/system/`: the systemd unit, the GPU tuning, the eGPU udev rule and the kernel command line (see its README).
- Paths under `/home/yuya` and `/mnt/win` are this machine's.

## What changed on top of upstream

- ROCm: the DeepSeek engine's HIP build on Strata 0.2.0 for gfx1100 (RX 7900 XTX) and gfx1201 (R9700); MXFP4 experts
  on the multi-entry expert kernels; IQ3_S as a grouped DOWN format.
- Expert tiers: several peer GPUs as expert stores filled from the file tier (`STRATA_PEER_GIB`,
  `STRATA_PEER_FILL=file`), and the peers compute their experts' rows during prompt reads (4K prefill 81 -> 140
  tokens/s); a copy of the model file on a second drive serves part of every file-tier read (`STRATA_BLOB_TWIN`,
  `STRATA_BLOB_MIRROR_SHARE`); file-tier blobs uploaded in chunks as their parts land; the arena misses' DMAs issued
  before the file-tier wait (+12-14 % decode).
- Decode: a fused decode layer (`GGML_OP_DS41_FUSED`); no HIP graph replay for the multi-row pass graphs; the DSpark
  drafter on a chosen GPU (`STRATA_DSPARK_DEVICE`). Upstream's lookup decoding and DSpark confidence cut are on in the
  production config.
- Long context: a 1M-token context (context-aware prompt-batch sizing; a 512K-token needle test passes).
- Serving, for an agent harness that interleaves conversations and subagents: conversation parks, tail resume,
  shared-prefix parks (optionally kept as files), END parks, request slots, STOP during a prompt read. The commit
  `DS41 serve: ...` describes each and its environment variables.
- Models and tools: unsloth's DeepSeek-V4.1-Flash-1017 GGUFs and split model files load; converters for the Lucebox
  drafter (DSpark sidecar) and its routing statistics (expert rank); route counts of the real traffic
  (`STRATA_ROUTE_COUNTS`, `tools/v41/rank_from_counts.py`); a residency-aware expert splice (`tools/v41/splice_b.py`,
  evaluated, not adopted); an Engram hot set built from a local corpus.

## Measured on this machine

- Quality on held-out text against the FP8 model: perplexity 8.63 (FP8: 8.56), top-1 agreement 93.6 %, KL 0.045
  (`local/eval-lucebox/run-eval-3gpu.sh`).
- Decode on real agent traffic (2026-10-22): 7.6-8.5 tokens/s; ~10.7 tokens/s on copy-heavy output with lookup
  decoding.
- Prompt reading: a 6,600-token prompt in ~29 s. A resent prompt resumes in 1.5-3 s and a switch between parked
  conversations takes 4-8 s, instead of a full re-read (~8 min at 190K tokens).

## Not in this repository

Model files, logs, evaluation outputs, the API key (`lan.env`; see the example in `local/system/`), the agent harness's
setup and data, the network setup, and `local/resume-test/` (it replays a private session).
