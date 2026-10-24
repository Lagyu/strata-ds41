# Model comparisons, 2026-10-23..25

Three published builds of DeepSeek-V4.1-Flash-1017 measured on this machine against the production model of the time
("MXFP4+ABL": the official checkpoint's MXFP4 experts, bit-exact, with Q8_0 dense layers carrying our abliteration of 62
dense tensors) and against the same model without the abliteration (the original weights). Outcome: since 2026-10-25
production runs orcarouter's Mix-256G with `--skip-file 0.10` (`local/ds41-yuya-3gpu.json`); the MXFP4+ABL config is the
rollback.

| Build | held-out ppl / top-1 / KL vs FP8 | security set KL / top-1 | decode tok/s: code / prose / ja / after 4K |
|---|---|---|---|
| original weights, no abliteration (MXFP4 experts) | 8.601 / 94.3% / 0.037 | the reference | - |
| MXFP4+ABL (production until 2026-10-25) | 8.626 / 93.6% / 0.045 | 0.020 / 97.3% | 9.0 / 9.0 / 10.5 / 7.0 |
| MXFP4+ABL + `--skip-file 0.10` | 8.627 / 91.5% / 0.059 | 0.017 / 97.6% vs MXFP4+ABL | 9.9 / 9.8 / 12.1 / 7.4 |
| orcarouter Mix-256G | 8.818 / 88.4% / 0.092 | 0.036 / 96.2% (0.038 / 96.1% vs MXFP4+ABL) | 12.4 / 11.7 / 13.9 / 9.9 |
| **Mix-256G + `--skip-file 0.10` (production since 2026-10-25)** | 8.835 / 87.6% / 0.100 | - | 13.0 / 12.3 / 14.6 / 10.1 |
| peasantsmith Maya-S + our abliterated dense layers | 9.587 / 81.8% / 0.217 | 0.089 / 93.9% vs MXFP4+ABL | 15.1 / 13.9 / 14.7 / 12.7 (*) |
| ISTA-DASLab GSQ-RCO IQ2_XS + our abliterated dense layers | 9.638 / 80.3% / 0.234 | 0.108 / 93.1% vs MXFP4+ABL | 11.4 / 17.0 / 19.4 / 14.7 (*) |

(*) one pass right after the service start, no warm-up; production in the same windows: 9.7 / 9.3 / 10.7 / 7.2 (Maya-S)
and 9.7 / 9.4 / 8.8 / 7.0 (GSQ-RCO). The Mix-256G and MXFP4+ABL rows are warm medians (below). 4K-token prompt reading:
Mix-256G ~236 tokens/s, MXFP4+ABL 105-168.

Refusal check (4 benign prompts that aligned models often refuse, thinking off): the original weights answered 2 of 4;
MXFP4+ABL, Mix-256G, Maya-S and GSQ-RCO with our abliteration answered 4 of 4.

Mix-256G's own abliteration also edits the routed experts' down weights (ours: 62 dense tensors), so its distance from the
original mixes abliteration and quantization; their unquantized abliterated model is not public.

## Method

- Held-out text: 8 x 1024 tokens, logits against the FP8 model's (`local/eval-lucebox/score.py`: perplexity, top-1
  agreement, KL on the reference's top 256), exact routing unless `--skip-file` is named (`local/maya-s/run-eval-mayas.sh`).
- Security code set: 16 x 2048 held-out tokens of security code (`local/secprof/build.py`), scored against the original
  weights' logits and against production's (`run-eval-sec-multi.sh`).
- The original weights: the internal-SSD copy of the production file with its 62 abliterated tensors switched back to the
  original bytes (`tools/orca/twin_noabl.py`, hash-checked both ways; the experts are the checkpoint's either way).
- Speed: `tools/common/bench.py` against the running service (thinking off, greedy): code / prose / Japanese prompts with
  256 output tokens and a 4K-token prompt with 128. Per config: service start, [one cold pass], a warm-up (12 varied
  prompts x 256 tokens: page cache, arena, VRAM cache, graphs, drafter), then 3 or 2 measured passes; the table shows the
  medians of the passes after the first. Every config ran on the production binary with the same drafter and flags, no
  route counts and no prefix files; MXFP4+ABL had its internal-SSD twin, Mix-256G has none (split files). Production
  measured again at the end of the window: 10.0 / 9.9 / 12.1 / 7.4 (no drift). Cold vs warm: decode within 10%, 4K
  prompt reading 30-40% slower right after a start.
- Refusals: `tools/common/refusal_check.py`.

## Files

- `chain-orca.sh PART` + `orca-lib.sh`: the Mix-256G windows (part 1: kernel preflight, held-out, security set, refusal
  check; part 2: speed), each started on an idle server (engine log untouched for 15 minutes); production is stopped in
  the window and restored by a trap. Configs `ds41-yuya-3gpu-{prodbench,prodbench-skip,mix256,mix256-skip,stock}.json`.
- `chain-qtest.sh` + `gsq-iq2xs.env` + `ds41-yuya-3gpu-gsq-iq2xs.json`: the GSQ-RCO window; `../maya-s/`: the Maya-S one.
- `run-eval-sec-multi.sh`: the security set scored against several references.
- `tools/`: copies of the scripts that ran from `/home/yuya/ds41` (their working copies on this machine; absolute paths
  kept). `common/`: bench.py, refusal_check.py, table.py and convert_ds41_mxfp4.py (the MXFP4 build tool, whose GGUF code
  the others import). `orca/`: download, `convert_orca.py` (their GGUF to the engine's layout: experts byte for byte,
  Q5_K/Q6_K dense to Q8_0, written as 3 split files of ~80 GB because ntfs3 cannot hold one 238 GB file), expert kernel
  tests, `twin_noabl.py`, `report_orca.py`. `gsq-rco/`, `maya-s/`: download, `merge_*.py` (their experts + our dense
  layers), kernel tests.
- `results/`: the three reports, `scores.json` (every score), `speed-rows.jsonl` (the bench rows without the generated
  text), `kernel-tests.txt`.

Model files are not in git: Mix-256G in `/mnt/win/llm/orca/` (production), MXFP4+ABL in `/mnt/win/llm/` plus its
internal-SSD twin (the rollback), the downloads and the other test builds in `/mnt/ext/models/`.
