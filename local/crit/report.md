# Prefill + decode critical paths (strata-ds41, 2026-10-30 evening)

Work dir /home/yuya/ds41/crit (window scripts, arms1/ arms2/ = probe outputs + engine logs, build logs); code on branch
critpath (worktree /home/yuya/llm-src/strata-zcnvis); private ggml source /home/yuya/llm-src/zcnvis/llama.cpp (the matrix-core
indexer + the fused hc kernels), archives ggml-prebuilt-crit / -crit2, binaries build-hip-deploy/ds41_generate-{wmma,c2}.
Production untouched so far (binary c1da353d, config e8b04daa9).

## 1. Where the time goes (measured)

Production log, 1356 turns: prompt-side 43459 s = batched reads 30970 (dense 10120 + experts 20383) + cache re-open 9154 +
tail passes 3335; decode: see the morning report (/home/yuya/ds41/decprof/report.md).

Window 1 (production code + profilers, /home/yuya/ds41/crit/arms1):

- A batched prompt read's expert phase is NVMe-bound: of its wall time 60-70% is "waits on reads" at 8.7-9.4 GB/s (A0 6600
  tokens: 9.5 s = reads 6.5 + final sync 1.4 + other host 1.5; GPU kernels 1.7 s underneath).  Both drives together top out
  near 10 GB/s (fio: SN8100 6.7 at 4 jobs, P41 3.4).  The arena DMA and the MMQ kernels hide under the reads.
- The dense phase is GPU-bound (sync wait 5.3 of 6.7 s at 6600 tokens; graph queueing 0.13 s) and slows with context:
  1000 new tokens cost dense 1.35 s at 8K context, 3.29 s at 98K (304 tok/s); 2500 new: 3.5 s -> 8.2 s.  The kernel trace
  (rocprofv3, first 127 s of a 98K read, <= 65K context) puts the scalar lightning indexer (lightning_indexer_kernel_vec) at
  17% of the GPU time = the single largest kernel, the WMMA attention (fa512) 15%, the peers' grouped expert kernels 23% (on
  the peers, in parallel), MMQ experts 12%, a hipBLAS SGEMM of M = 24 (the hc mixes, 1040 calls x 0.9 ms per 8192 batch)
  9.6%, the fused hc tail 5.5% (1.7 ms a 512-column span: a serial 20-iteration Sinkhorn on thread 0 of all 80 blocks a token).
  Per index-source layer and 512-query span at 98K the vector kernel runs ~125 ms (3.3 TFLOPS effective); 8 index layers.
- Engram rows: 0.8 s of the 6.7 s dense phase at 6600 tokens (12%; 4% at 3000).
- Decode pass (prose, 1.5-1.8 rows): 112 ms = attention+router graphs 38 (GPU 0.8 ms a layer for ~30 kernel dispatches; host
  queueing 0.07 ms, readback 0.07) + tier 50-74 (PCIe DMA-bound) + finish 4 + draft 6 + logits 1.3 + begin 0.5.
- Cache re-open per batched turn 0.7-1.0 s = the seed's "file jobs" (STRATA_SEED_PROF): 8 reader threads each
  cudaMallocHost + cudaFreeHost a 15 MB bounce per seed for 11-150 blobs - the allocation, not the reads.

## 2. Decode: the async layer sequence (STRATA_TIER_ASYNC) - no gain, not deployed

Implemented (Ds41MoeTier::set_async / done_event, Ds41Dense::attn_router_begin/end, finish_async, finish_predict_multi_async,
predict_wait; the tier records an event instead of its final stream sync, the finish graph waits it on ggml's stream, the next
layer's attention graph is queued before the host waits for the router ids).  Bit-identical by construction.  Measured
(A_std / A_sec / A_long vs B_*): prose 112.4 -> 115.2 ms/pass, code 140.3 -> 146.4, long-context 102 -> 112: a loss of 2-5%.
The host side was never the decode bottleneck - the attention graph's 0.8 ms a layer is GPU dispatch time for ~30 small
kernels and the tier is DMA-bound - and the later prefetch issue costs a little.  Kept as an opt-in switch, default off.

## 3. Prefill: the matrix-core lightning indexer (STRATA_INDEXER_WMMA) and the fused hc kernels

HIP ran the lightning indexer's scalar kernel (the WMMA one is CUDA-only).  New kernel (lightning-indexer.cu, namespace
lidx_wmma): 16x16x16 bf16 WMMA on gfx12 (f16 on gfx11), Q tile in LDS, each warp two 16-key tiles x the head tiles, relu-w
sum, the nvis / -INF mask paths of the vector kernel, MXFP4 and F16 keys.  Unit test (test_ds41_indexer --large,
LIDX_DUMP/LIDX_TIME): every engine shape 256..65536 keys x 512 queries passes, top-512 sets identical to the vector kernel
(max abs score diff 5.7e-6), 13x faster at 262K keys (window 2 T1).  Engine (W_long2, -80 mV,
clean): the 98K read 156.1 -> 111.9 s (dense 94.8 -> 61.5 s, -35%; experts 52.3 -> 46.8), +150 tokens at 98K 5.82 -> 3.89 s
(dense 0.60 -> 0.28), +1000 9.84 -> 6.95 (dense 3.31 -> 1.31), +2500 16.19 -> 10.16 (dense 8.29 -> 3.35), the 3000-token
bench 18.52 -> 11.57 s (dense 10.15 -> 4.06): at 98K context the dense phase costs 2.5x less and a turn costs about what it
costs at 8K context.  The first attempt faulted at -100 mV (section 6).  Held-out eval (window 2 HO, 8 x 1024 tokens, STRATA_INDEXER_WMMA=1 alone): KL
0.10055 / top-1 87.65% / PPL 8.873 against zcnvis-deploy's 0.10137 / 87.90% / 8.851 (s027-FA2: 87.79%) - within the eval's
noise (the kernel's bf16 scores can swap keys at the top-512 margin).

The fused hc kernels (ds41-fused.cu): the hc-tail prep (one thread per token instead of thread 0 of every block, bit-exact)
and a fused rms_norm + hc-mixes kernel replacing the M = 24 hipBLAS SGEMM per span (DS41F_HC_MIXES, spans >= 128 columns,
last-ulp class; STRATA_HC_MIXES_FUSED, default on).  Measured C2_std vs B2_std (window 2): the dense phase -6..10% at short
context (A0 6600 tokens: dense 6.29 -> 5.63 s, the turn 16.58 -> 14.80; +2500: 3.28 -> 3.06; bench_W 3.35 -> 3.10; bench_S1
2.67 -> 2.51), experts unchanged; C2_long: PENDING.

## 4. Prefill: the chunk-prefetch slab (STRATA_CHUNK_PF_MIN / _FRAC, two-drive reader) - a loss, not deployed

The existing whole-layer prefetch (the next layer's file tier read into a pinned slab under this layer's expert phase) was
made usable (the budget --chunk-prefetch N GiB honoured, its reader split over both drives in 8 parts, peer-held experts
excluded, a partial slab STRATA_CHUNK_PF_FRAC) and measured in window 2 (c2 binary, /home/yuya/ds41/crit/arms2):

- whole tier (PF_long, MIN 2048, 8 GiB): the 98K read ran far slower (killed at the 30-minute cap; 145 s without); the slab's
  16 reader threads and the wave reader share the two drives, and the slab reads the whole next layer while this layer's
  wave reads only what the batch routes.
- half tier (PF5, MIN 512, FRAC 0.5, 4 GiB): long context +1000 tokens 9.84 -> 11.35 s (+15%), +2500 16.19 -> 15.62, the 98K
  read 156 -> 139 s (-11%, but see the dense noise below); short context (PF5_std vs B2_std): A0 6600 tokens 16.58 -> 16.0 s,
  +1000 6.91 -> 6.9, +2500 10.23 -> 9.99, bench_W 10.98 -> 10.49 (-2..4%).  The chunk profile shows why it cannot win big:
  the batch's "waits on reads" fall (4.8 -> 2.0 s) but the "final sync" rises (1.3 -> 2.4 s): with the reads hidden, the
  phase is bounded by the PCIe DMA of every streamed expert (arena + file, ~155 GB per 6600-token batch at 27 GB/s) and the
  grouped kernels, and the slab's own reads slow the wave reader (22 GB at 5.6 GB/s instead of 49 GB at 8.6).

Verdict: not deployed.  The routing-hinted stage (section 4b) replaces it: it reads only what the batch routes, as soon as
each 512-column span has routed, under the same layer's dense half.

### 4b. The routing-hinted file-tier reads (STRATA_CHUNK_HINT_GIB) - window 3

Ds41MoeTier::chunk_hint_begin(layer) before a layer's dense spans, chunk_hint(layer, ids, wts, n) after each span's router:
a pinned slab (4 GiB = ~270 expert slots, a layer's whole routed file tier) read by 32 persistent threads x 8 parts over both
drives; run_chunk DMAs the hinted experts out of the slab (an atomic wait on the slot's ready flag if the last part is still
coming) and reads only the rest through the wave reader.  Only routed experts are read, the skip rule's dropped experts
(weights below 0.10 of the row for every token of the span) excluded as run_chunk excludes them: exact.  Exactness (window 3 SM_off vs SM_hint:
a 16K one-shot prefill + 24 greedy tokens with and without the stage): the outputs are byte-identical; 8227 file-tier experts
were hinted over the 2 batches and all 8227 of run_chunk's file-tier DMAs came out of the slab (none through the wave reader).
Timing at short context (window 3, same-window baseline B3_std): the 6600-token read's expert phase 7.87 -> 7.43 s
(-6%) but its dense half 5.97 -> 6.25 (the hint's 9 GB/s of reads slow the Engram row reads that the dense half makes
from the same drive), the 771-token copy task's expert phase 5.39 -> 6.06 (+12%: run_chunk streams the file-tier experts
in its own order while the slab fills in hint order, so a small batch waits head-of-line), the other turns within noise.
The chunk profile shows the mechanism: "waits on reads" 4.8 -> 2.5 s a batch, "final sync" 1.3 -> 3.7 s - once the reads
are hidden the phase sits on the PCIe DMA floor (section 6b).  Net at short context: neutral.  Long context (H4_long, 98K,
fused kernels off, vs window 2's B2_long / C2_long): the expert phase of a +1000 turn 5.57 / 4.76 -> 4.83 s, +2500 6.68 / 5.80
-> 5.52, the 3000-token bench 7.38 / 6.54 -> 5.96 (-5..10% of the expert phase vs C2), the 98K read's 52.3 / 47.1 -> 48.3
(noise).  The stage still waited 1.6-2.3 s per turn on hinted reads (chunk hint: "waited"): run_chunk streams the file tier
in ascending expert order while the slab filled in token order, so a batch blocks on an expert queued behind ones it does
not need yet.  Fixed in c4 (each span's new experts are enqueued in ascending id order) - but c4's reorder had a dedup bug (a claimed
expert was re-queued once per routing token, the slab overflowed, every expert phase doubled: window 4 is void; the outputs
stayed identical), fixed in c5 (a08a970df).  Window 5 (c5, outputs still identical): short context H5_std vs B3_std - the 6600-token read's
expert phase 7.87 -> 7.51 but its dense half 5.97 -> 7.04, the copy task's expert phase 5.39 -> 5.96, the rest within noise;
ALL5_std vs ALL_std: the same picture (copy task 7.58 vs the baseline's 7.13); the stage still waited 2-3.6 s per turn on
its own reads (at short context the dense half is shorter than the layer's reads).  98K context, ALL5_long vs W_long2 (the
indexer alone): the read 109.8 vs 111.9 s, +1000 6.84 vs 6.95, +2500 9.84 vs 10.16, the 3000-token bench 10.96 vs 11.57
(-3..5%, the expert phase -0.2..0.5 s).  Three cache re-opens of 1.7-5.3 s appeared only in hint arms (the 4 GiB pinned slab
shrinks the page cache the seed's file-tier reads live in; not proven).

VERDICT: not deployed.  With skip-read and the queued chunk sum on, the file-tier reads are no longer what bounds the
expert phase - the PCIe 4.0 link is (6b) - so hiding them buys little, and the stage costs small batches 6-10%, the dense
half at short context, and 4 GiB of pinned RAM.  It stays an exact opt-in (STRATA_CHUNK_HINT_GIB) worth re-measuring behind
a PCIe 5.0 root port or with the Engram file on a drive of its own.

The skip-read exclusion (STRATA_CHUNK_SKIP_READ, default on, c2 binary): file-tier experts the skip rule drops for every
token of the pass are no longer read or streamed - 580-900 experts (9-13 GB) per batch at short context (chunk prof "left
unread").  Exact (their weights were already zero).  Its effect is measured by SK0_std (window 3): PENDING.

## 5. Per-turn fixed costs: the cache re-open (seed buffers), the Engram hot pack

The cache re-open after a batched turn (0.7-1.5 s here, p50 2.1 s in production = 21% of its prompt-side time) is the
seed: STRATA_SEED_PROF splits it into cache open, queue, "file jobs", fills, peers.  The 8 x cudaMallocHost/FreeHost per seed
were replaced by persistent bounce buffers (kept, harmless) - no change: the "file jobs" time is the ~16 GB arena -> VRAM
seed DMA (~1050 slots x 15 MB at 27 GB/s) queued ahead of the file blobs' blocking copies.  Keeping the cache across a batch
turn was measured on 2026-10-24 (keep-cache, no net gain at a full card) and is not revisited here.
The Engram hot pack (STRATA_ENGRAM_HOT, HOT_std vs B2_std): the dense phase -10% on the 6600-token read (6.29 -> 5.64 s,
the Engram wait was 12% there), neutral at <= 3000 tokens, +4 GB of anonymous RSS (the hot rows are pinned in RAM for the
process lifetime) on an engine already at 118 GB RSS of 125 - not deployed.

A partial seed (STRATA_SEED_FRAC=0.5, SF5_std: seed half the slots, let the decode's arena-held misses take the free slots)
cut the re-open to 0.36-0.42 s (from 0.72-0.99) but PRODUCED WRONG TOKENS: 1.00 token per pass, no lookup / draft
acceptance, ~100% VRAM hits (a degenerate routing), 39 of 40 tokens different from the baseline in every turn (the other
arms differ from the baseline only by late near-tie divergences).  The queued fills are synced on stream 0 before the
decode starts, so the race I first suspected is not it; the cause is not found yet.  The switch stays off and MUST NOT be
set until it is; it is not in the candidate.

## 6. GPU faults during the windows (R9700 undervolt)

Two R9700 faults at the start of a 98K layer-major read, both with the -100 mV offset applied (r9700-undervolt.service):
22:23 under rocprofv3 (vector indexer): amdgpu gfxhub PAGE FAULT in mul_mat_q<Q8_0> (a dense matmul), "GCVM_L2_PROTECTION_FAULT
client TCP"; 23:21 with the matrix-core indexer: "unspecified launch failure", then at teardown "MES(0) failed to respond to
msg=REMOVE_QUEUE ... MODE1 reset" - the 10-29 hang's signature.  The reset dropped the offset to 0 mV (card1 OD_VDDGFX_OFFSET
0mV since 23:22:47); every run after that (one-shot 16K, serve warm+16K+append, window 2) ran clean.  The matrix-core
kernels draw more power than the scalar ones.

Third fault, 2026-10-31 01:24:57, at -80 mV (the user's new persistent setting), arm B3_long of window 3 = the c3 binary
with the fused kernels OFF and WITHOUT the matrix-core indexer, no hint - production's own kernels: "unspecified launch
failure" at the start of the 98K layer-major read, then "MES(0) failed to respond to msg=REMOVE_QUEUE ... MODE1 reset"
(dmesg 15337 s; "device wedged, but recovered through reset", the second such reset since boot).  The reset dropped the
offset to 0 mV again (OD_VDDGFX_OFFSET 0mV while r9700-undervolt.service still reports active); every later run (ALL_long,
HO_all, window 4) ran at stock voltage.  Three faults at -100 / -100 / -80 mV, all at the same point (the first batches of a
98K read: the dense half at full 8192-column batches plus the expert streaming = the card's heaviest sustained load), none
at 0 mV across windows 2-4: the R9700 is not stable undervolted under this workload.  The card is at 0 mV now and I did not
re-apply the offset; /etc/r9700-undervolt.conf still says -80 and would re-apply at the next boot.  Recommendation: set
voltage_offset_mv = 0 (or disable the service); the XTX peers' -110 mV were never implicated.

## 6b. The PCIe ceiling is the board's root port (hardware)

dmabench (window 3, R9700, pinned host -> device): 26.9 GB/s on one stream, 27.2 on two, 27.7 on four; 1 MB chunks 22.7-26.0;
HSA_ENABLE_SDMA=0 (blit kernels instead of the SDMA engines) 26.8; device -> host 27.9.  No stream count or copy engine
changes it: the ceiling is the link.  lspci: the R9700's own switch ports run 32 GT/s x16 internally, but its upstream port
reports "Speed 16GT/s (downgraded)" against the CPU root port 00:01.0, whose LnkCap2 lists 2.5-16 GT/s only - the ASRock
B760 Pro RS/D4 WiFi (BIOS 10.03) exposes the i7-14700K's x16 port as PCIe 4.0 (the port negotiated Gen4 at every
level; amdgpu's pp_dpm_pcie shows 16 GT/s at all three states).  27 GB/s is PCIe 4.0 x16's practical rate.  Everything
PCIe-bound here (the decode tier's ~1.5 ms a layer of expert DMA, the batch expert phase's ~155 GB of streaming per
6600-token batch, the 16 GB cache re-seed per batched turn) would run up to ~2x faster behind a PCIe 5.0 x16 root port.
This is a platform limit, not a software one: a board whose CPU slot is PCIe 5.0 (or confirming in the BIOS that PCIE1 is
not forced to Gen4, if the board's slot supports Gen5 at all) is the only lever.

## 7. Recommendation / deploy

DEPLOYED 2026-10-31 02:13 (deploy-crit.sh): production binary = the c3 build (a70df23a: critpath 74e6d3cbd + the private ggml
archive crit2), config env + STRATA_INDEXER_WMMA=1 + STRATA_CHUNK_ASYNC=1; the fused hc kernels and the skip-read exclusion
are on by default in the binary; the hint stage, the half seed, the slab prefetch and the two copy streams are off.  Smoke
"17 + 25" -> 42; the serve log shows "indexer: RDNA matrix-core kernel".  Rollback: local/ds41-yuya-3gpu.json.prod-1030d +
build-hip/ds41_generate.prod-1030d (c1da353d) + /home/yuya/dsh-work/tools/restart-ds41.sh.  ds41-0.2.0 fast-forwarded to
critpath; the config change committed on top.  The hint stage stays off (window 5, section 4b).  Final production state verified 02:30: service active, API 200, smoke
"17 + 25" -> 42, engine a70df23a with STRATA_INDEXER_WMMA=1 STRATA_CHUNK_ASYNC=1.

What production gained (window measurements vs the pre-window production code, same payloads):
- long context (98K): a 98K read 156 -> ~110 s; +1000 tokens 9.8 -> 6.9 s; +2500 16.2 -> 9.8-10.2 s; a 3000-token turn
  18.5 -> 11-11.6 s (the dense phase 2.5x faster: the matrix-core indexer).
- short context (8-11K): batched turns -5..10% (6600-token read 16.6 -> 14.1-14.3 s; 2500-3000-token turns 10-11 -> 9.5-10.5 s),
  security payload -2..21% per turn; small batches and the 771-token copy task unchanged; decode unchanged.
- quality: held-out KL 0.1006 / top-1 87.6% vs 0.1014 / 87.9% before (within the eval's noise); the hint, skip-read and the
  queued chunk sum are exact by construction, the indexer selects the same top-512 keys, the fused hc-mixes kernel is last-ulp.

Not deployed, measured and documented: the async decode tier (2), the chunk-prefetch slab (4), the Engram hot pack and the
half seed (5; the half seed produces wrong tokens and must not be set), two copy streams (6b), the hint stage (4b).
Hardware findings for the user: the CPU's x16 slot is PCIe 4.0 on this board (6b: the ceiling of everything PCIe-bound), and
the R9700 faulted at -80 mV as it did at -100 (6: recommend 0 mV).

Candidate (window 3 "ALL", binary ds41_generate-c3, env on top of production's): STRATA_INDEXER_WMMA=1 (section 3),
STRATA_CHUNK_HINT_GIB=4 (4b), STRATA_CHUNK_ASYNC=1 (the next layer's reads start ~30 ms/layer earlier: A0 16.6 -> 15.2 s,
the rest neutral), the fused hc kernels and the skip-read exclusion (default on in the binary).  Not in the candidate: the
chunk-prefetch slab (4), STRATA_TIER_ASYNC (2), the Engram hot pack (dense -10% on a 6600-token read only, neutral below
3000 tokens, +4 GB of anonymous RSS on an engine already at 118 of 125 GB).  Measured separately in window 3: the half
seed (STRATA_SEED_FRAC=0.5: WRONG TOKENS, off), two copy streams (STRATA_CHUNK_DMA_STREAMS=2: flat, as the 27 GB/s link
predicts), skip-read off (SK0_std vs B3_std: +0.8-1.1 s on every batched turn = the exclusion saves 7-10%).  Deploy gate: the
held-out eval of the candidate (HO_all) within noise of zcnvis-deploy (KL 0.10137 / top-1 87.90%; the hint, skip-read and
chunk-async are exact by construction, the matrix-core indexer selects the same top-512 keys, the fused hc-mixes kernel is
last-ulp).  Window 3 results so far (c3 binary a70df23a, same-window baseline B3_std unless noted):

- ALL_std vs B3_std: 6600-token read 15.10 -> 14.08 s (-7%; dense 5.97 -> 5.26), +2500 10.06 -> 9.58, bench 3000 10.96 ->
  10.48, bench 2500 9.80 -> 9.47 (-3..5%); the 771-token copy task 7.13 -> 7.75 (+9%, the hint's head-of-line effect);
  two one-off stalls (A1's cache re-open 2.3 s, A3's dense 3.7 s) not seen in any other arm.  Dense phases -12..18%.
- ALL_sec vs window 1's B_sec (production code): 16.20 -> 15.43, 5.20 -> 4.09, 5.73 -> 5.18, 7.84 -> 7.14, 11.49 -> 10.02,
  12.07 -> 11.86, 10.91 -> 10.03, 8.49 -> 8.29 s (-2..21%); decode speed unchanged.
- SK0_std (skip-read off): +0.8-1.1 s on every batched turn -> the exclusion is worth 7-10%.
- D2_std (two copy streams): flat.  SF5_std (half seed): wrong tokens, off (section 5).
- ALL_long (98K context): the read 118.2 s (B2_long 156.1, C2_long 133.0, W_long2 111.9: the matrix-core indexer is the
  whole story there; dense 62.5 s), +400 5.36 s (C2 5.84), +1000 7.21 (8.72), +2500 9.98 (14.70), the 3000-token bench
  11.86 (16.99).  B3_long (the same-window long baseline) exited early (see below); window 2's B2_long / C2_long stand in.
- HO_all (the deploy gate, full candidate): KL 0.10059 / top-1 87.59% / PPL 8.873 against zcnvis-deploy's 0.10137 / 87.90% /
  8.851 and the indexer-only run's 0.10055 / 87.65% - the fused hc-mixes kernel adds nothing measurable; PASSED.

## 8. After the deploy: the live workload (numeric log lines) and the candidates prepared for the next window

The production log's per-turn summary lines (no content), the new binary's segment (02:26-05:17, 40 turns, one session at
night) against the previous binary's last segments (2026-10-30 evening, 183 turns, several sessions - a confounded comparison):

| new tokens | turns | now: median wall | feed (dense + experts) | cache re-open | DSpark tail | before: median wall |
|---|---|---|---|---|---|---|
| <= 96 (tail passes only) | 10 | 2.2 s | - | (warm-up turns only) | 2.0 s | 2.8-3.1 s |
| 97-500 | 12 | 4.2 s | 3.0 (1.5 + 1.5) | 0.67 s | 0.49 s | 9.1 s |
| 501-2000 | 8 | 6.2 s | 5.9 (2.9 + 2.9) | 0.83 s | - | 17.5 s |
| 2001-8000 | 9 | 11.2 s | 11.5 (6.0 + 5.2) | 0.85 s | - | 27.5 s |

Decode: median 18.1 tok/s per request (p25 17.2, p75 20.1), expert routes 74.5% VRAM hits / 24.1% PCIe / 1.3% file.  Short
turns are the majority of the user's turns (25% here, 80% in the older history) and consist of 16-row DSpark tail passes
(~0.7 s a pass, PCIe-bound).  In a batched turn the dense half and the expert phase are about equal, and the cache
re-open is 7-16% of the turn.

Prepared (implemented, built, NOT deployed, NOT measured - production was serving the user; window6.sh runs them):

- **STRATA_CHUNK_HINT_ARENA=1 (c7)** with STRATA_PRESTAGE_GIB=4: as each 512-column dense span routes, its arena-held
  experts (not VRAM-resident, not peer-held) are DMA'd into the existing layer-prestage region on the copy stream while
  the dense half runs; run_chunk launches them from VRAM (the existing "layer prestage" path, MMVQ kernels) and streams
  only the rest.  The arena experts are ~2/3 of the expert phase's PCIe traffic (6b), so up to that share of the
  phase can hide under the dense half; exact (same bytes, the eager / lazy prestage stand down).  The region's VRAM
  (4 GiB) comes out of the batch's free VRAM; the OOM hook releases it if a pass does not fit.
- **STRATA_SEED_PARTS=8 x STRATA_SEED_THREADS=32 (c6, default on in c6/c7; =1 is the old reader)**: the re-seed's
  file-tier blobs read in 8 parts by a 32-thread pool over both drives (read_part's twin share) instead of 8 whole-blob
  readers on the model drive; exact.  Aims at the "file jobs" part of the 0.7-0.85 s re-open.
- **STRATA_SERVE_KEEP_CACHE=1** (existing, off): continuation turns keep the VRAM cache across the batch (shrunk by the
  batch's needs) - no re-seed, VRAM hits during the batch.  Measured 2026-10-24 as a wash on a different profile; the
  live profile (re-open 7-16% of a batched turn, the expert phase at the PCIe floor) says re-measure it.
- Evaluated and not pursued: wider DSpark tail passes (kMaxMulti 32): the user's short turns average ~42 new tokens, so
  two wider passes cost about what three narrow ones do; the gain appears only above ~90 tokens.  Engram reads are
  already parallel, coalesced and cached (2M rows); the hot pack is the only lever and costs 4 GB of RAM.

## 9. Window 6 / 7 (2026-10-31 09:51-): the prepared candidates measured

Production was stopped for this (dsh-web stopped at the user's request); the baseline B6 already runs with the user's new
`--batch-slots 4` (the restart at the window's end applies it to production).

- **Arena hint, first form (HA_std, c7)**: exact (the one-shot's 24 tokens identical; 9664 experts into the 4 GiB region, none
  left out) but 15-25% SLOWER on every batched turn (6600-token read 14.48 -> 17.51 s, expert phase 8.21 -> 10.62).  The
  region-served experts went through the layer-prestage launch path = the grouped (decode-style) kernels, and the host side
  of the expert phase grew to 2.8 s a batch.  Reworked (c8, window 7): the region's experts are copied device-to-device into
  the MMQ staging slots - the same kernels and run grouping as a streamed expert, the PCIe copy replaced by a ~500 GB/s one.
- **Seed parts (SEEDP_std, c7)**: the engine died at the first re-seed with "malloc(): unaligned tcache chunk detected" (heap
  corruption in the part reader).  STRATA_SEED_PARTS defaults back to 1; opt-in until the bug is found.  Not deployed.
- **Keep-cache (KC_std)**: never engaged - "ExpertCache::swap_slots: the slots are sized per layer (uniform slots only)": the
  Mix-256G cache opens with per-class slot sizes and the shrink needs uniform slots.  Numbers identical to B6.  Not applicable
  without a sized-slot shrink.
- B6_std (deployed set, 4 slots): 6600-token read 14.48 s, +1000 6.50, +2500 9.53, 3000-token bench 10.52, copy task 6.93 -
  the same as window 3's 2-slot numbers within noise.  B6_long (4 slots): the 98K read 107.7 s (2 slots: 109.8-111.9), +1000
  6.79, +2500 10.02, the 3000-token bench 11.27 - no 4-slot penalty visible on this payload.
- The three combined arms (ALL6_*) ran the crashing seed reader and HUNG to the 30-minute cap each (the engine did not exit);
  window 6 therefore ended at 11:44 instead of ~10:35.

**Arena hint, re-routed (HA2_std, c8, window 7)**: exact again (identical one-shot tokens; 5288-5927 arena experts a batch copied
device-to-device out of the region).  Big read (6600 tokens): the expert phase 8.21 -> 7.68 s but the dense half 5.08 -> 5.70 -
the hint's ~90 GB of PCIe copies under the dense half slow the dense half's own small transfers; the batch's read waits
stay 5.4 s (49 GB of file-tier reads at 8.7 GB/s were the longer pole at this size, not the arena copies).  Small and
medium turns: +150 tokens 3.69 -> 4.81 s, +400 4.84 -> 6.32, +1000 6.50 -> 7.91, the copy task 6.93 -> 8.63 - a dense half of
0.2-1.1 s cannot hide 1.5 GB of arena copies a layer, and the copies queue ahead of the expert phase's own stream copies.
VERDICT: not deployed in either form; the idea would need to be conditional on wide batches and would still cost the dense
half.  The file-tier reads, not the arena copies, bound the expert phase of big reads on this box.
