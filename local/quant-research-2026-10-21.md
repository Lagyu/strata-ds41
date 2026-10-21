# Mixed-precision quantization for DS41 on this box: the options (2026-10-21)

Requested before the 1M-token runs: "research the ways of quantization; mixed precision like Xolotl can help us run
faster while keeping most of the quality."  Everything below is measured on this box unless marked as someone
else's number.

## 1. Where decode time goes today, and why expert bytes are the lever

Production file `/mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf` (278 GiB):

| part | bytes | format |
|---|---|---|
| routed experts: 40 layers x 384 = 15,360 experts, 17.93 MiB each | 268.95 GiB | MXFP4 (the checkpoint's own blocks, lossless) |
| attention / dense / head | 7.38 GiB | Q8_0 (201 tensors) + F16/F32 |
| shared experts | 1.40 GiB | Q8_0 |
| Engram | from the ROCMFP2S file via `--engram` | |

Quality on our harness (held-out 8x1024, vs the FP8 teacher): ppl 8.626, top-1 93.56 %, KL 0.0448.

Residency with the service config (95 GiB arena, R9700 28 GiB + XTX2 20 GiB peer stores filled from the file tier):

| tier | experts | cost per use |
|---|---|---|
| peers (R9700 1,599 + XTX2 1,142) | 2,741 | none (computed in place) |
| card 0 cache (opens after a prompt) | ~500-600 | none |
| pinned-RAM arena | 5,425 | 17.9 MiB over x16 (~0.75 ms at 24 GB/s) |
| NVMe file tier only | 7,194 | 17.9 MiB from the P41 + twin (~2-3 ms) |

Decode traffic, 256-token requests at 8.9-10.2 tok/s (service log):

| request | hit | arena DMA | NVMe read | bytes / token |
|---|---|---|---|---|
| prose 9.71 tok/s | 236 / token (80 %) | 51 (17 %) | 9 (3 %) | 1.05 GiB |
| code 8.87 tok/s | 158 (62 %) | 86 (34 %) | 11 (4 %) | 1.70 GiB |

At ~24 GB/s (x16 DMA) and ~8 GB/s (P41 + twin) that is 60-90 ms of a ~105 ms token; the dense half is ~20 ms.
Decode is bound by the bytes of NON-resident experts, and the hit rate is set by how many experts fit on the
peers and in the arena.  Quantizing experts attacks both: fewer bytes per fetch, and 1.4-2.1x more experts resident.
The prompt phase streams every expert once per layer-major batch (221 GiB of arena + file per batch), so it gains
similarly (the 512K needle spent 1,597 of 2,761 s in experts).

## 2. What Xolotl's XQ / GSQ does (IndrasMirror)

- Per-expert GPTQ on calibration activations (128 x 2048 tokens: real agent transcripts + 4,690 imatrix prompts),
  static Hessian-weighted scales, three tiers: PTQ1_0 ternary 1.75 bpw, Q2_0 2.25 bpw, Q3_K 3.44 bpw.
- RCO-lite allocation: Lagrangian relaxation under a byte budget, cost = held-out output error of each expert at
  each tier (weight-only error cannot rank FP4-native experts, so calibration is required), guard rails on the
  ternary fraction and on unseen experts.  Dense Q8_0, embeddings/indexer F16, norms/router F32.
- 1017 v2 mix: 62.4 % ternary / 24.7 % Q2_0 / 13.0 % Q3_K = 2.09 bpw (142 GiB); v3 = 152 GiB (~2.3 bpw), calibrated
  40 % code / 30 % agentic.  Their quality vs their edited teacher: v3 ppl 6.72 vs 6.36 (+5.7 %), top-1 0.691 vs
  0.706 (-1.5 pt).  Their speed (RTX 4090, 90 GB RAM): exact decode 12-13 tok/s, with DSpark 23-26 tok/s.
- Uncensoring: the same mrfakename-style rank-2 refusal-direction edit as ours (attention output + shared-expert
  down), applied at full precision before quantizing.  Engram not included (we keep our `--engram` file).
- The whole pipeline is in our tree (tools/v41: v41_calib / v41_gptq / v41_alloc / v41_quant / v41_gguf) and our
  engine IS Xolotl's fork: per-expert format classes (up to 9), kernels for MXFP4, Q2_K, Q3_K, Q2_0, PTQ1_0, IQ1_S/M,
  IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS.  But the pipeline reads the HF safetensors (FP8 dense + FP4 experts + the two
  101.5 GB Engram shards) through DeepSeek's torch model.py; we have neither the safetensors nor a ROCm torch.
  Their compute estimate: calibration forward a few hours + GPTQ 15-40 GPU-hours (4090).

## 3. Ready-made files

| file | size | expert formats | quality (their numbers) | notes |
|---|---|---|---|---|
| IndrasMirror XQ-1017-v3 Uncensored | 163 GB (152 GiB), 16 parts + experts.json + rank-1017-v3.bin | GPTQ ternary / Q2_0 / Q3_K per expert; Q8_0 dense | ppl +5.7 %, top-1 -1.5 pt vs their teacher; our KL unknown | loads in our engine; same refusal edit as ours; 43 MB/s anonymous = ~65 min |
| unsloth UD-IQ3_S | 262 GB (+161 GB engram, not needed) | IQ3_S mix, imatrix | KL 0.0316 / 0.0195 (wiki1100 / long17k vs F32), top-1 96.0 / 96.8 % | barely smaller than our 269 GiB |
| unsloth UD-IQ3_XXS | 233 GB | IQ3_XXS mix | KL 0.0625 / 0.043, top-1 94.6 / 95.1 % | stock (censored): `patch_gguf.py apply-edit` works on its Q8_0 dense |
| unsloth UD-Q2_K_XL | 200 GB | Q2_K mix | KL 0.138 / 0.088, top-1 91.5 / 93.3 % | |
| unsloth UD-IQ2_XXS | 165 GB | IQ2_XXS mix | KL 0.203 / 0.142, top-1 90.5 / 91.2 % | |
| unsloth imatrix (789 MB) | downloaded to /home/yuya/models/unsloth-1017/imatrix/ | | | lets us requantize our own file with their calibration |

Scale caveat: unsloth's KL is against an F32 reference on their sets; on OUR harness their UD-Q4_K_XL scored
0.036 (their table: 0.009 / 0.0045), so our numbers run ~4x higher for the same file.  Our 0.0448 is Q4-class on
their scale.  Any candidate must be scored on our harness before comparing with 0.0448.

## 4. The residency-aware idea (what I would build)

Only non-resident experts cost bytes, and usage is very skewed (the 2,741 peer experts + card-0 cache serve 62-80 %
of fetches).  So: keep the hot experts (the peer set) lossless MXFP4, quantize the warm set (arena) and the cold set
(file tier) to small formats.  The quality loss is confined to experts that serve ~20-35 % of fetches, the byte
saving lands exactly where the time is, and smaller cold experts also pull more of them into RAM:

| plan | cold bytes | arena holds | file tier left | est. decode | est. our KL |
|---|---|---|---|---|---|
| today | 17.9 MiB | 5,425 | 7,194 | 9.5 tok/s | 0.0448 |
| B: hot MXFP4 + warm IQ3_XXS (12.9 MiB) + cold IQ2_XXS (8.7 MiB) | -28 % / -52 % | 7,540 | 5,079 | ~12-14 | ~0.05-0.06 |
| B2: hot MXFP4 + rest IQ2_XXS | -52 % | 11,180 | 1,439 (<1 % of fetches) | ~15-17 | ~0.07-0.09 |
| C: uniform IQ3_XXS (unsloth-style) | -28 % everywhere | 7,540 | 4,020 (peers hold 3,800) | ~13-15 | ~0.06 |
| A: XQ v3 as is (~2.3 bpw GPTQ everywhere) | -50 % everywhere | all 15,360 resident (143 GiB = 48 peer + 95 arena) | 0 | ~18-22 | unknown (guess 0.10-0.15) |
| D: own GPTQ mixed, hot MXFP4 kept (Xolotl pipeline on a GGUF shim) | -50 % cold | like B2 | ~15-17 | ~0.06 | |

The tok/s columns are estimates from the byte model above (token time ~ dense 20 ms + bytes / bandwidth + sync);
the first real build will calibrate them.

Effort:
- A: ~2-3 h, no code: download (65 min), reassemble, `--rank` to rank-1017-v3.bin, load, run the eval harness
  (run-eval-3gpu.sh) and the speed set.  The cheap bracket: the speed ceiling of ~2 bpw on this box and a real KL.
- C: ~1 day: a requantizer (gguf-py + ggml's quantize via ctypes, unsloth's imatrix, MXFP4 -> IQ3_XXS per tensor),
  no engine change, same eval.  Or 1.5 h download of UD-IQ3_XXS + the refusal patch on its dense.
- B / B2: C's requantizer + a per-expert-class writer (Xolotl's XQ layout, experts.json) that takes each expert's
  bytes from the MXFP4 file or the requantized one by the rank file, + a loader check that MXFP4 is accepted as a
  class.  2-3 days.  The split (how many hot, which cold format) is a parameter, so B and B2 are one tool.
- D: ~1 week + 20-40 GPU-hours: a GGUF-backed replacement for the safetensors reader in v41_calib (our file has the
  identical FP4 expert blocks), ROCm torch, DeepSeek's reference model.py.  Best quality per byte at ~2 bpw; builds
  on B's writer.

## 5. Recommendation

1. A now (reversible, no GPU time until the eval): it answers "how fast can this box go" and "is 2.3 bpw GPTQ
   acceptable" with real numbers in an afternoon.
2. B's tooling in parallel; first build = B (IQ3_XXS warm / IQ2_XXS cold), then B2 if the KL budget allows.
   This is the quality-first answer: the experts that serve most tokens stay lossless.
3. D only if B's RTN-with-imatrix cold experts cost more KL than wanted; it reuses B's writer.

The long runs (1M MRCR rows, the 1,040,000-token needle) stay on hold until you choose.
