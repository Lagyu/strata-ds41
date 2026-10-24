// tools/v41/engine/ds41_generate.cpp - the Strata DeepSeek-V4.1-Flash engine end to end: Ds41Dense (attention,
// Engram, router, shared expert; CUDA or CPU ggml backend) + Ds41MoeTier (routed experts with per-expert formats:
// VRAM cache, pinned RAM arena, file tier, CPU pool, PCIe share) + sampling.  Token ids in, token ids out (text is
// the server's job: tools/v41/serve/serve_ds41.py).
//
//   ds41_generate -m V41-GSQ.gguf (--ids 1,2,3 | --ids-file f.i32) [-n 64] [--backend cuda|cpu] [--experts gpu|cpu]
//                 [--engram DIR:DIR:DIR] [--engram-hot hot.bin] [--slots auto|N] [--vram-margin GiB] [--arena-gib 60] [--rank r.bin]
//                 [--vram-lru] [--arena-adapt] [--arena-skip-vram] [--pcie 0.25] [--tier-batch-pcie 1.0] [--prefill-chunk N] [--prompt-batch B] [--chunk-mmq] [--ctx N]
//                 [--temp 0] [--seed 1] [--stop id,id] [--ppl] [--no-actquant] [--kv-fp4] [--serve] [--quiet]
//                 [--enc-skip|--no-enc-skip] [--enc-replay N] [--dump-last F] [--dump-routes DIR]
// --enc-skip (layer-major batches only; DEFAULT for one-shot runs, --no-enc-skip = full path): prompt tokens before the last --enc-replay (default 2560 = 128 * 20) stop
// after layer 20's persistent-KV writes; only the last 2560 tokens run layers 21-39.  Same last-token logits / decode.
//
// --serve speaks Strata's engine line protocol (~/AI/Strata/serve/server.py StrataEngine, as glm_generate):
//   in:  GEN <max_new> [temperature=.. top_p=.. top_k=.. min_p=.. seed=..] <ids csv> | STOP | QUIT
//        GENI <max_new> [k=v ..] <image embeddings file> <ids csv>   (GEN with vision; see --image-token)
//   out: READY <ctx> stop, RESUME <reused>, PP <read> <total> <ms> <tok/s>, T <id>, DONE ..., ERR <text>
// Conversation reuse: a request that extends what the state has seen continues; one that shares the last prompt
// restores the snapshot taken at its end (Ds41Dense::snapshot: every other cache is position-indexed); the same prompt
// again (regenerate) restores it and samples from the saved last-row logits without a prefill.  VRAM: a multi-token
// prompt runs with the expert cache closed (its buffers need that VRAM) and the decode after it re-seeds the cache from
// the measured free memory (~0.6 s), as a one-shot run; the decode biases (--route-bias / --route-bias-file) apply only
// while decoding.
//
// Load the real model only through tools/ds4/memguard.sh (RAM cap + the shared GPU lock).
#include "ds41_dense.hpp"
#include "ds41_moe.hpp"
#if defined(DS41_HAVE_DSPARK)
#include "ds41_dspark.hpp"
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <functional>
#include <memory>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---- repetition penalties: the state has to be declared before the argument parsing that sets it ----------
struct PenaltyState {
    float repeat  = 1.0f;   // penalty_repeat: positive logits divided, negative multiplied (llama.cpp rule)
    float freq    = 0.0f;   // penalty_freq: subtract freq * occurrences
    float present = 0.0f;   // penalty_present: subtract once if the token is in the window
    int   last_n  = 0;      // penalty_last_n: window over the last N emitted tokens (0 = off)
    int   cap     = 4096;   // like the platform's kPenaltyWindowCap
    std::vector<int> hist;  // emitted tokens, oldest first
};
static PenaltyState g_pen;

// ---- routing counts (--dump-routes DIR): how many tokens selected each (layer, expert), in the format
// tools/v41/engine/v41_rank.py reads (DIR/LNN.json, experts[e].n = tokens routed to e).  The router's `ids` are
// already a graph output (Ds41Dense::Impl::o_ids) and already copied host-ward per layer for the expert tier
// (attn_router's readback, batch_ids() after attn_router_span), so counting here changes no arithmetic.
struct RouteCounts {
    std::string dir;
    int n_layer = 0, n_experts = 0, top_k = 0;
    std::vector<uint64_t> cnt;   // [n_layer][n_experts]
    bool on() const { return !dir.empty(); }
    void init(const std::string & d, int nl, int ne, int tk) {
        dir = d; n_layer = nl; n_experts = ne; top_k = tk;
        if (on()) cnt.assign((size_t) nl * (size_t) ne, 0);
    }
    /// --serve: start from the counts already in `dir` (a previous server's), so the counts add up across restarts.
    /// Reads the format write() produces ("\"e\": {\"n\": N}"); a missing or unreadable file is just zeros.
    uint64_t load() {
        if (!on()) return 0;
        uint64_t total = 0;
        for (int l = 0; l < n_layer; ++l) {
            char name[64];
            std::snprintf(name, sizeof(name), "L%02d.json", l);
            std::ifstream f(dir + "/" + name);
            if (!f) continue;
            const std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            const size_t ex = txt.find("\"experts\"");
            if (ex == std::string::npos) continue;
            for (size_t at = ex; (at = txt.find("{\"n\": ", at)) != std::string::npos;) {
                size_t q1 = txt.rfind('"', at);          // the closing quote of the expert's key
                size_t q0 = q1 == std::string::npos ? q1 : txt.rfind('"', q1 - 1);
                at += 7;
                if (q0 == std::string::npos) continue;
                const int e = std::atoi(txt.c_str() + q0 + 1);
                const uint64_t n = std::strtoull(txt.c_str() + at, nullptr, 10);
                if (e >= 0 && e < n_experts) { cnt[(size_t) l * (size_t) n_experts + (size_t) e] += n; total += n; }
            }
        }
        return total;
    }
    /// Count n tokens' routing for layer l (ids: n * top_k ints, token-major).
    void add(int l, const int * ids, int n) {
        if (!on() || !ids || l < 0 || l >= n_layer || n <= 0) return;
        uint64_t * row = cnt.data() + (size_t) l * (size_t) n_experts;
        for (int t = 0; t < n; ++t)
            for (int j = 0; j < top_k; ++j) {
                const int e = ids[(size_t) t * top_k + j];
                if (e >= 0 && e < n_experts) ++row[e];
            }
    }
    bool write(std::string & err) const;
};
static RouteCounts g_routes;

bool RouteCounts::write(std::string & err) const {
    if (!on()) return true;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) { err = "cannot create " + dir + ": " + ec.message(); return false; }
    uint64_t total = 0;
    for (int l = 0; l < n_layer; ++l) {
        char name[64];
        std::snprintf(name, sizeof(name), "L%02d.json", l);
        std::ofstream f(dir + "/" + name);
        if (!f) { err = "cannot write " + dir + "/" + name; return false; }
        const uint64_t * row = cnt.data() + (size_t) l * (size_t) n_experts;
        f << "{\"layer\": " << l << ", \"top_k\": " << top_k << ", \"experts\": {";
        for (int e = 0; e < n_experts; ++e) {
            f << (e ? ", " : "") << "\"" << e << "\": {\"n\": " << row[e] << "}";
            total += row[e];
        }
        f << "}}\n";
    }
    const double tokens = (double) total / std::max(1, n_layer * top_k);
    std::fprintf(stderr, "routes: %d layers x %d experts, %.0f tokens routed (%.1f selections/expert) -> %s\n",
                 n_layer, n_experts, tokens, (double) total / (double) std::max<size_t>(1, (size_t) n_layer * n_experts), dir.c_str());
    return true;
}

using namespace strata;
using namespace strata::v41;

namespace {

struct Args {
    std::string model, ids_csv, ids_file, rank, dump_logits, dump_last, dump_routes;
    std::string dense_bench;      // --dense-bench R:P: after the prompt, P verify-shaped passes of R rows with the expert tier
                                  // skipped (routed sums = 0) - the dense half's own time per pass / layer, no NVMe or PCIe
                                  // noise; the rows' logits go to --dump-last (all P*R rows) for a bit-exact A/B of dense
                                  // kernel changes.  Garbage text by construction.
    std::string build_mirror;      // --build-mirror F [--mirror-gib N]: write the blob mirror (coldest experts by --rank) and exit
    double mirror_gib = 85.0;
    std::string prune;            // --prune F: "layer expert" lines (coldest first - v41_prune_list.py), soft or hard pruned
    float prune_penalty = 0.0f;   // --prune-penalty X: selection bias -X on the listed experts (0 = masked out of routing)
    float route_bias_file = 0.6f;    // --route-bias-file X: selection penalty -X on file-tier experts (neither VRAM nor RAM arena);
                                     // DEFAULT 0.6 (Mal: max decode w/o too much loss): 21.1 vs 15.1 tok/s, code ppl 2.67->2.87, agentic 6.88->6.39
    int   bias_prompt = 0;          // --bias-prompt N: quality proxy - apply route bias during the prompt, treating the first N
                                    // ranked experts as VRAM-resident (what the deferred seed will cache)
    float route_bias = 1.0f;      // --route-bias X: selection bonus +X on the experts resident in VRAM at the seed (DEFAULT 1.0, Mal 2026-10-11:
                                  // ~4x decode; 4096-tok ppl code 2.38 -> 2.67, agentic 15.1 -> 6.9; --route-bias 0 = exact routing)
    std::vector<std::string> engram;
    std::string engram_hot;
    int n_predict = 64;
    std::string backend = "cuda", experts = "gpu";
    int64_t slots = 0;            // 0 = auto (free VRAM after the dense half, minus --vram-margin)
    int dslots = 0;               // --serve: --batch-slots N (conversation slots on the device; 0 = STRATA_SERVE_DSLOTS or 1)
    // defaults = the measured best decode config on the 4090 (2026-10-11): --pcie 0.75 --pf-b 1.43 --vram-lru
    double vram_margin_gib = 1.0, pcie = 0.75, arena_gib = 60.0, pf_b = 1.43;
    double tier_batch_pcie = -1.0;   // < 0 = STRATA_TIER_BATCH_PCIE / 1.0
    int threads = 0, prefill_chunk = 0, prompt_batch = 8192;   // prompt_batch: layer-major batch B (0 = the chunk path)
    int64_t ctx = 0;
    float temp = 0.0f;
    uint64_t seed = 1;
    std::vector<int> stop;
    bool ppl = false, serve = false, quiet = false, vram_lru = true, arena_adapt = false, chunk_mmq = true;
    bool act_quant = true, chunk_prefetch = true, kv_fp4 = false, arena_skip_vram = false;
    /// --peer-device N / STRATA_PEER_DEVICE=N: the SECOND GPU's expert tier (docs/ds4/V41_AMD_PEER_TIER.md); -1 =
    /// off, in which case nothing about the tier changes.  N >= 0 must be a CUDA/HIP device index that is not the
    /// dense half's (0 on this box): its memory holds the ranked experts after card 0's VRAM head, and a decode
    /// token's misses that it holds are computed there (no blob moves), in parallel with the primary's kernels.
    std::vector<int> peer_devices = [] {   // --peer-device 1[,2,...] / STRATA_PEER_DEVICE: the other cards' expert tiers
        std::vector<int> v;
        if (const char* e = std::getenv("STRATA_PEER_DEVICE"))
            for (const char* q = e; *q;) { v.push_back(std::atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
        return v;
    }();
    int  kv_type = -1;            // --kv-q8 / --kv-type N: the compressed KV + indexer keys' ggml type (-1 = BF16 with act_quant, else F32)
    int  enc_skip = -1;           // -1 = default (on for one-shot layer-major runs), 1 = --enc-skip, 0 = --no-enc-skip; only run the tokens before the replay window through layers 21-39
    int  enc_replay = 2560;       // --enc-replay N: the replay window (window_size * (n_layer - cand_layer) = 128*20)
    bool enc_bounded = false;     // --enc-bounded: accept --enc-replay below the exact reach, the decoder window clipped at the
                                  // replay start (DeepInfer's SWA bounded replay, its V4.1 default is 140 = 128 + 12); approximate
    float skip_miss = 0.0f, skip_file = 0.0f;
    double chunk_prefetch_gib = 0.0;
    int image_token = 129264;     // --image-token N: the model's <|image|> id (each image is one run of it in the prompt)
    bool vision = false;
};

std::vector<int> parse_csv(const std::string & s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) if (!x.empty()) v.push_back(std::atoi(x.c_str()));
    return v;
}

// The image-embedding file of a GENI line: one or more strata-vision records (int32 'SVE1', n, nx, ny, n_embd, then
// n * n_embd float32) in prompt order (serve/server.py combined_embeddings_path; generate.cpp reads the same).
static bool read_sve1(const std::string & path, std::vector<std::vector<float>> & rows, std::vector<int64_t> & counts,
                      int64_t n_embd, int64_t max_rows, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open the embeddings file"; return false; }
    int32_t hdr[5];
    int64_t total = 0;
    for (;;) {
        f.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
        if (f.gcount() == 0 && f.eof()) break;
        if (f.gcount() != sizeof(hdr)) { err = "short embeddings header"; return false; }
        if (hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 || hdr[4] != (int32_t) n_embd
            || (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[1] > max_rows - total) {
            err = "bad embeddings file (expected strata-vision records of width " + std::to_string(n_embd) + ")";
            return false;
        }
        const int64_t n = hdr[1];
        total += n;
        std::vector<float> r((size_t) (n * n_embd));
        if (f.read(reinterpret_cast<char *>(r.data()), (std::streamsize) (r.size() * 4)).gcount()
                != (std::streamsize) (r.size() * 4)) { err = "short embeddings file"; return false; }
        rows.push_back(std::move(r));
        counts.push_back(n);
    }
    if (rows.empty()) { err = "the embeddings file has no records"; return false; }
    return true;
}

// Map records onto image-token runs, using the row counts to split adjacent images within the same run.
static bool map_vision(const std::vector<int> & ids, int image_token, const std::vector<std::vector<float>> & rows,
                       const std::vector<int64_t> & counts, std::vector<float> & vis_rows,
                       std::vector<int64_t> & vis_pos, std::string & err) {
    size_t k = 0;
    for (size_t i = 0; i < ids.size(); ) {
        if (ids[i] != image_token) { ++i; continue; }
        size_t j = i;
        while (j < ids.size() && ids[j] == image_token) ++j;
        const int64_t run = (int64_t) (j - i);
        if (k >= rows.size()) { err = "the prompt has more images than the embeddings file"; return false; }
        int64_t matched = 0;
        while (matched < run && k < rows.size()) {
            if (counts[k] > run - matched) { err = "an image span does not match its embeddings"; return false; }
            matched += counts[k];
            vis_rows.insert(vis_rows.end(), rows[k].begin(), rows[k].end());
            ++k;
        }
        if (matched != run) { err = "an image span is missing embeddings"; return false; }
        for (size_t t = i; t < j; ++t) vis_pos.push_back((int64_t) t);
        i = j;
    }
    if (k != rows.size()) { err = "the embeddings file has more images than the prompt"; return false; }
    return true;
}

bool parse(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (k == "-m" || k == "--model") a.model = next();
        else if (k == "--ids") a.ids_csv = next();
        else if (k == "--ids-file") a.ids_file = next();
        else if (k == "-n" || k == "--n-predict") a.n_predict = std::atoi(next().c_str());
        else if (k == "--backend") a.backend = next();
        else if (k == "--experts") a.experts = next();
        else if (k == "--peer-device") {
            a.peer_devices.clear();
            const std::string v = next();
            for (const char* q = v.c_str(); *q;) { a.peer_devices.push_back(std::atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
        }
        else if (k == "--engram-hot") a.engram_hot = next();
#ifdef _WIN32
        // the Windows path-list separator: a drive letter's ':' must not split "D:\a;E:\b"
        else if (k == "--engram") { std::stringstream ss(next()); std::string d; while (std::getline(ss, d, ';')) a.engram.push_back(d); }
#else
        else if (k == "--engram") { std::stringstream ss(next()); std::string d; while (std::getline(ss, d, ':')) a.engram.push_back(d); }
#endif
        else if (k == "--slots") { const std::string v = next(); a.slots = v == "auto" ? 0 : std::atoll(v.c_str()); }
        else if (k == "--batch-slots" || k == "--batch") a.dslots = std::atoi(next().c_str());
        else if (k == "--vram-margin") a.vram_margin_gib = std::atof(next().c_str());
        else if (k == "--arena-gib") a.arena_gib = std::atof(next().c_str());
        else if (k == "--rank") a.rank = next();
        else if (k == "--dump-routes") a.dump_routes = next();   // calibration: per (layer, expert) routing counts
        else if (k == "--build-mirror") a.build_mirror = next();
        else if (k == "--check-mirror") { a.build_mirror = next(); a.mirror_gib = -1; }
        else if (k == "--mirror-gib") a.mirror_gib = std::atof(next().c_str());
        else if (k == "--vram-lru") a.vram_lru = true;
        else if (k == "--no-vram-lru") a.vram_lru = false;
        else if (k == "--arena-adapt") a.arena_adapt = true;
        // --arena-skip-vram: the arena holds the ranking PAST the VRAM cache's head instead of duplicating it (see
        // Ds41MoeConfig::arena_skip_resident): with a rank this buys ~1.9 GiB of distinct experts per 1.5 GiB of slots
        else if (k == "--arena-skip-vram") a.arena_skip_vram = true;
        else if (k == "--pcie") a.pcie = std::atof(next().c_str());
        else if (k == "--tier-batch-pcie") a.tier_batch_pcie = std::atof(next().c_str());
        else if (k == "--pf-b") a.pf_b = std::atof(next().c_str());
        else if (k == "--prefill-chunk") a.prefill_chunk = std::atoi(next().c_str());
        else if (k == "--prompt-batch") a.prompt_batch = std::atoi(next().c_str());
        // prompt experts on MMQ (int8 tensor cores; PTQ1_0 through its lossless Q2_0 transcode) - DEFAULT since
        // 2026-10-12: 8K prompt 219-220 -> 244-248 tok/s, 4096 code ppl 2.3799 -> 2.3790, KL 7.6e-3, deterministic
        // (8K --dump-last md5 ee31507a0e88; the MMVQ path's was 4d229102872a).  --no-chunk-mmq = the MMVQ kernels.
        else if (k == "--chunk-mmq") a.chunk_mmq = true;
        else if (k == "--no-chunk-mmq") a.chunk_mmq = false;
        else if (k == "--no-chunk-prefetch") a.chunk_prefetch = false;
        else if (k == "--chunk-prefetch") a.chunk_prefetch_gib = std::atof(next().c_str());
        else if (k == "--skip-miss") a.skip_miss = (float) std::atof(next().c_str());
        else if (k == "--skip-file") a.skip_file = (float) std::atof(next().c_str());
        else if (k == "--threads") a.threads = std::atoi(next().c_str());
        else if (k == "--ctx") a.ctx = std::atoll(next().c_str());
        else if (k == "--temp") a.temp = (float) std::atof(next().c_str());
        else if (k == "--penalty-repeat") g_pen.repeat = (float) std::atof(next().c_str());
        else if (k == "--penalty-freq") g_pen.freq = (float) std::atof(next().c_str());
        else if (k == "--penalty-present") g_pen.present = (float) std::atof(next().c_str());
        else if (k == "--penalty-last-n") g_pen.last_n = std::max(0, std::atoi(next().c_str()));
        else if (k == "--seed") a.seed = (uint64_t) std::atoll(next().c_str());
        else if (k == "--stop") a.stop = parse_csv(next());
        else if (k == "--ppl") a.ppl = true;
        else if (k == "--no-actquant") a.act_quant = false;
        else if (k == "--kv-fp4") a.kv_fp4 = true;
        // 8-bit cache: the same compressed-KV / indexer-key tensors as Q8_0 instead of the default BF16 (act_quant).
        // ggml's flash-attn takes Q8_0 K/V (its kv-type list), set_rows quantizes on the way in and the ring's
        // ggml_cast dequantizes on the way out, so this halves the state's bytes at 8 bits a weight - the 4-bit
        // grids (--kv-fp4) are the aggressive ones and need the packers of tools/v41/ggml/kv-fp4.patch.
        else if (k == "--kv-q8") a.kv_type = (int) GGML_TYPE_Q8_0;
        else if (k == "--kv-type") a.kv_type = std::atoi(next().c_str());
        else if (k == "--serve") a.serve = true;
        else if (k == "--prune") a.prune = next();
        else if (k == "--prune-penalty") a.prune_penalty = (float) std::atof(next().c_str());
        else if (k == "--route-bias") a.route_bias = (float) std::atof(next().c_str());
        else if (k == "--bias-prompt") a.bias_prompt = std::atoi(next().c_str());
        else if (k == "--route-bias-file") a.route_bias_file = (float) std::atof(next().c_str());
        else if (k == "--quiet") a.quiet = true;
        else if (k == "--dump-logits") a.dump_logits = next();
        else if (k == "--dump-last") a.dump_last = next();
        else if (k == "--dense-bench") a.dense_bench = next();
        else if (k == "--enc-skip") a.enc_skip = 1;
        else if (k == "--enc-bounded") a.enc_bounded = true;
        else if (k == "--no-enc-skip") a.enc_skip = 0;
        else if (k == "--enc-replay") a.enc_replay = std::atoi(next().c_str());
        else if (k == "--image-token") a.image_token = std::atoi(next().c_str());
        else if (k == "--vision") a.vision = true;
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return false; }
    }
    return !a.model.empty() && (a.serve || !a.ids_csv.empty() || !a.ids_file.empty() || !a.build_mirror.empty());
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- repetition penalties (design: docs/ds4/V41_SAMPLER_PENALTIES.md) -----------------------------------------
// This engine had no penalty stage; every other stack has one (the platform this fork comes from:
// penalty_repeat/freq/present/last_n, generate.cpp:10648; llama.cpp: --repeat-penalty/--frequency-penalty/
// --presence-penalty).  The defaults below are the platform's and make this a no-op: the logits are then untouched
// and the generated tokens are bit-identical to before.  Applied inside both samplers, so the speculative path's
// draft (sample_dist) and verify (sample_p) sample the same penalised row.

static void pen_record(int tok) {
    if (g_pen.last_n <= 0) return;
    g_pen.hist.push_back(tok);
    if ((int) g_pen.hist.size() > g_pen.cap) g_pen.hist.erase(g_pen.hist.begin());
}

// The buffer to sample from: the input itself when there is nothing to do, else a penalised copy in scratch.
static const float * pen_apply(const float * logits, int n_vocab, std::vector<float> & scratch) {
    if (g_pen.last_n <= 0) return logits;
    if (g_pen.repeat == 1.0f && g_pen.freq == 0.0f && g_pen.present == 0.0f) return logits;
    const int n = (int) g_pen.hist.size();
    const int from = n > g_pen.last_n ? n - g_pen.last_n : 0;
    if (from >= n) return logits;
    scratch.assign(logits, logits + n_vocab);
    std::vector<int> count((size_t) n_vocab, 0);
    for (int i = from; i < n; ++i) {
        const int t = g_pen.hist[(size_t) i];
        if (t >= 0 && t < n_vocab) count[(size_t) t]++;
    }
    for (int t = 0; t < n_vocab; ++t) {
        const int c = count[(size_t) t];
        if (c == 0) continue;
        float & l = scratch[(size_t) t];
        if (g_pen.repeat != 1.0f) l = l > 0.0f ? l / g_pen.repeat : l * g_pen.repeat;
        if (g_pen.freq != 0.0f) l -= g_pen.freq * (float) c;
        if (g_pen.present != 0.0f) l -= g_pen.present;
    }
    return scratch.data();
}

// temperature + top-k / min-p / top-p over the top candidates (top_k 0 = the 256 best); temp 0 = argmax
static int sample_p_raw(const float * logits, int n_vocab, float temp, int top_k, float top_p, float min_p, std::mt19937_64 & rng) {
    if (temp <= 0.0f) return (int) (std::max_element(logits, logits + n_vocab) - logits);
    const int K = std::min(n_vocab, top_k > 0 ? top_k : 256);
    std::vector<int> idx((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
    const double mx = logits[idx[0]];
    std::vector<double> p((size_t) K);
    for (int i = 0; i < K; ++i) p[(size_t) i] = std::exp(((double) logits[idx[(size_t) i]] - mx) / temp);
    int n = K;
    if (min_p > 0.0f) while (n > 1 && p[(size_t) n - 1] < (double) min_p * p[0]) --n;
    if (top_p > 0.0f && top_p < 1.0f) {
        double zn = 0, c = 0;
        for (int i = 0; i < n; ++i) zn += p[(size_t) i];
        for (int i = 0; i < n; ++i) { c += p[(size_t) i] / zn; if (c >= top_p) { n = i + 1; break; } }
    }
    double z = 0;
    for (int i = 0; i < n; ++i) z += p[(size_t) i];
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng);
    for (int i = 0; i < n; ++i) if ((r -= p[(size_t) i]) <= 0) return idx[(size_t) i];
    return idx[(size_t) n - 1];
}

// records the emitted token, so the window is the sequence actually produced
static int sample_p(const float * logits, int n_vocab, float temp, int top_k, float top_p, float min_p, std::mt19937_64 & rng) {
    std::vector<float> scratch;
    // pen_record belongs where the sequence grows, not here: a draft the verify accepts never passes through
    // sample_p, and recording here would count every other token twice.
    return sample_p_raw(pen_apply(logits, n_vocab, scratch), n_vocab, temp, top_k, top_p, min_p, rng);
}

// the distribution sample_p draws from, as (token, probability) pairs summing to 1 (temp 0: the argmax alone) - the
// trunk's p for --serve's speculative sampling (same candidates, same weights as sample_p, and the same penalised
// row: a draft is not an emitted token, so nothing is recorded here)
void sample_dist(const float * logits_in, int n_vocab, float temp, int top_k, float top_p, float min_p,
                 std::vector<std::pair<int, double>> & out) {
    std::vector<float> scratch;
    const float * logits = pen_apply(logits_in, n_vocab, scratch);   // shadows the parameter: the body is unchanged
    out.clear();
    if (temp <= 0.0f) { out.push_back({ (int) (std::max_element(logits, logits + n_vocab) - logits), 1.0 }); return; }
    const int K = std::min(n_vocab, top_k > 0 ? top_k : 256);
    std::vector<int> idx((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int x, int y) { return logits[x] > logits[y]; });
    const double mx = logits[idx[0]];
    std::vector<double> p((size_t) K);
    for (int i = 0; i < K; ++i) p[(size_t) i] = std::exp(((double) logits[idx[(size_t) i]] - mx) / temp);
    int n = K;
    if (min_p > 0.0f) while (n > 1 && p[(size_t) n - 1] < (double) min_p * p[0]) --n;
    if (top_p > 0.0f && top_p < 1.0f) {
        double zn = 0, c = 0;
        for (int i = 0; i < n; ++i) zn += p[(size_t) i];
        for (int i = 0; i < n; ++i) { c += p[(size_t) i] / zn; if (c >= top_p) { n = i + 1; break; } }
    }
    double z = 0;
    for (int i = 0; i < n; ++i) z += p[(size_t) i];
    for (int i = 0; i < n; ++i) out.push_back({ idx[(size_t) i], p[(size_t) i] / z });
}

// Speculative sampling with a DETERMINISTIC draft d (DSpark drafts by argmax: q = a point mass on d).  Leviathan /
// Chen: accept with min(1, p(d) / q(d)) = p(d); on rejection sample from norm(max(0, p - q)) = p without d.  The
// emitted token is then distributed exactly as p - the same as plain sampling.  true = d accepted; false = `out` is
// the replacement token.
bool spec_accept(const std::vector<std::pair<int, double>> & dist, int d, std::mt19937_64 & rng, int & out) {
    double pd = 0;
    for (const auto & c : dist) if (c.first == d) pd = c.second;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (pd > 0 && u(rng) < pd) return true;
    double r = u(rng) * (1.0 - pd);
    int last = -1;
    for (const auto & c : dist) {
        if (c.first == d) continue;
        last = c.first;
        if ((r -= c.second) <= 0) { out = c.first; return false; }
    }
    out = last >= 0 ? last : d;   // rounding: the last candidate (d only if it was the sole candidate - then pd was 1)
    return false;
}

double log_softmax_at(const float * lg, int nv, int tok) {
    const float mx = *std::max_element(lg, lg + nv);
    double z = 0;
    for (int i = 0; i < nv; ++i) z += std::exp((double) lg[i] - mx);
    return (double) lg[tok] - mx - std::log(z);
}

}  // namespace

int main(int argc, char ** argv) {
    // HIP graph replay of the multi-row pass graphs (DSpark verify passes, their finish + prediction) costs ~0.4 ms a
    // layer in a decode whose expert tier keeps other streams busy with DMAs (3 GPUs, measured 2026-10-21: 8.45 ->
    // 9.28 tok/s without it, the same bits; the dense bench alone shows no such cost).  ggml-cuda's
    // GGML_CUDA_GRAPH_MAX_TOKENS=1 keeps the replay for the 1-token graphs only; a value set by the caller wins.
    setenv("GGML_CUDA_GRAPH_MAX_TOKENS", "1", 0);
    if (!std::getenv("DS41_GGML_DEBUG"))
        ggml_log_set([](ggml_log_level lv, const char * text, void *) { if (lv != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr); }, nullptr);
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr, "usage: see tools/v41/engine/ds41_generate.cpp\n");
        return 2;
    }
    if (!a.build_mirror.empty()) {   // the second-drive blob mirror (STRATA_BLOB_MIRROR): coldest experts first
        std::string err;
        if (a.mirror_gib < 0) {   // --check-mirror F: open it with every blob compared against the model file
            ds41_sys::setenv("STRATA_BLOB_MIRROR", a.build_mirror.c_str(), 1);
            ds41_sys::setenv("STRATA_BLOB_MIRROR_CHECK", "0", 1);
            Ds41FileBlobs chk(a.model);
            if (!chk.open(err)) { std::fprintf(stderr, "ds41_generate: experts: %s\n", err.c_str()); return 1; }
            return chk.mirrored() > 0 ? 0 : 1;
        }
        Ds41FileBlobs src(a.model);
        if (!src.open(err)) { std::fprintf(stderr, "ds41_generate: experts: %s\n", err.c_str()); return 1; }
        if (a.rank.empty()) { std::fprintf(stderr, "ds41_generate: --build-mirror needs --rank\n"); return 2; }
        std::vector<std::pair<int32_t, int32_t>> order = ds41_read_ranked(a.rank, err);
        if (order.empty()) { std::fprintf(stderr, "ds41_generate: --rank: %s\n", err.c_str()); return 1; }
        std::reverse(order.begin(), order.end());
        if (!src.build_mirror(a.build_mirror, order, (uint64_t) (a.mirror_gib * 1073741824.0), err)) {
            std::fprintf(stderr, "ds41_generate: build-mirror: %s\n", err.c_str());
            return 1;
        }
        return 0;
    }
    std::vector<int> prompt = a.ids_csv.empty() ? std::vector<int>() : parse_csv(a.ids_csv);
    if (!a.ids_file.empty()) {
        std::ifstream f(a.ids_file, std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "ds41_generate: cannot open %s\n", a.ids_file.c_str()); return 2; }
        prompt.resize((size_t) f.tellg() / 4);
        f.seekg(0);
        f.read((char *) prompt.data(), (std::streamsize) (prompt.size() * 4));
    }
    std::mt19937_64 rng(a.seed);

    // ---- dense half
    ggml_backend_t be = a.backend == "cuda" ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr)
                                            : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) { std::fprintf(stderr, "ds41_generate: no %s backend\n", a.backend.c_str()); return 1; }
    if (ggml_backend_is_cpu(be)) ggml_backend_cpu_set_n_threads(be, a.threads > 0 ? a.threads : 8);
    int64_t ctx = a.ctx > 0 ? a.ctx : (int64_t) prompt.size() + a.n_predict + 64;
    ctx = std::max<int64_t>(4096, (ctx + 255) / 256 * 256);
    Ds41DenseConfig dc;
    dc.backend = be;
    dc.n_threads = a.threads > 0 ? a.threads : 8;
    dc.ctx_max = ctx;
    // the layer-major batch B: --prompt-batch, capped to the prompt length for a one-shot run (the state tensors hold
    // B columns, so a short prompt must not pay for the flag's full size; --serve sizes for the largest request)
    const int PB = a.prompt_batch > 0 ? (a.serve ? a.prompt_batch : (int) std::min<int64_t>(a.prompt_batch, (int64_t) prompt.size())) : 0;
    // the dense sub-batch size S: --prefill-chunk, or 512 with a layer-major prompt batch and no chunk size given
    const int S = a.prefill_chunk > 0 ? a.prefill_chunk : (a.prompt_batch > 0 ? 512 : 1);
    dc.max_pass = S;
    dc.prompt_batch = PB;
    // --serve: elastic state columns (STRATA_ELASTIC_COLS=0 = the fixed prompt_batch columns): decode keeps S columns and
    // the ~0.31 MiB a column of a big prompt batch (2.6 GiB at 8192) goes to VRAM expert slots; a layer-major batch
    // grows them (the VRAM cache is released for it anyway) and gives them back.  s11: 2.6 GiB of slots = 10-20% of a
    // verify pass (one-shot, margin 1.0 / 2.3 / 3.6: python 131 / 136 / 145 ms, code 89 / 93 / 107 ms).
    dc.elastic_cols = a.serve && PB > 1 && !(std::getenv("STRATA_ELASTIC_COLS") && std::getenv("STRATA_ELASTIC_COLS")[0] == '0');
    // --serve's conversation slots (--batch-slots N / --batch N, else STRATA_SERVE_DSLOTS; default 1 = the engine as before,
    // byte for byte): N conversation states on the device (Ds41DenseConfig::n_slots, Xolotl's batch slots - docs/BATCHING.md).
    // The requests running at once then keep their state resident instead of parking it in host memory at every turn of
    // the slice, and decode TOGETHER in batch windows (one row of each per pass).  Each slot costs its full-context state
    // (0.91 GiB at 1M with --kv-fp4), taken from the VRAM expert cache.
    if (a.serve) {
        int ds = a.dslots;
        if (ds <= 0 && std::getenv("STRATA_SERVE_DSLOTS")) ds = std::atoi(std::getenv("STRATA_SERVE_DSLOTS"));
        dc.n_slots = std::max(1, std::min(16, ds));
    }
    // encoder-only prompt prefill (enc-skip): tokens before the last `enc_replay` need their layers past layer 20 only
    // through the per-layer 128-token window, so for a long prompt they can stop after the KV-writing part of layer
    // 20 (the layer-major batch path only; the chunk path has no per-layer column split, --ppl/--dump-logits need
    // every row's logits, which the skipped rows do not have).
    // default ON (byte-identical last-row logits vs the full path, 8192 tokens, 2026-10-11) for layer-major runs, --serve
    // included (a continuation / snapshot restore only reads per-position caches + the last window of every layer,
    // which the replay window computes in full)
    bool enc_skip = a.enc_skip == 1 || (a.enc_skip < 0 && PB > 1);
    if (enc_skip && a.enc_replay < 1) { std::fprintf(stderr, "ds41_generate: --enc-replay must be >= 1\n"); return 2; }
    if (enc_skip && (a.ppl || !a.dump_logits.empty())) {
        if (a.enc_skip == 1) std::fprintf(stderr, "note: --enc-skip disabled (--ppl/--dump-logits need every row's logits, the old full path)\n");
        enc_skip = false;
    }
    if (enc_skip && PB <= 1) { std::fprintf(stderr, "ds41_generate: --enc-skip needs the layer-major prompt path (--prompt-batch >= 2)\n"); return 2; }
    int64_t ER = enc_skip ? (int64_t) a.enc_replay : 0;
    dc.engram_dirs = a.engram;
    dc.engram_hot = a.engram_hot;
    dc.vision = a.vision;
    dc.act_quant = a.act_quant;
    // --kv-fp4 / STRATA_KV_FP4=1: the compressed KV + indexer keys packed to their exact FP4 grids (Ds41DenseConfig::kv_fp4)
    dc.kv_fp4 = a.kv_fp4 || (std::getenv("STRATA_KV_FP4") && std::getenv("STRATA_KV_FP4")[0] == '1');
    if (dc.kv_fp4) std::fprintf(stderr, "kv: compressed KV NVFP4 + indexer keys MXFP4 (--kv-fp4, exact)\n");
    dc.kv_type = a.kv_type;
    if (a.kv_type >= 0) std::fprintf(stderr, "kv: compressed KV + indexer keys as %s (--kv-q8/--kv-type)\n",
                                     ggml_type_name((ggml_type) a.kv_type));
    Ds41Dense dense;
    std::string err;
    const double t_load = now_ms();
    if (!dense.init(a.model, dc, err)) { std::fprintf(stderr, "ds41_generate: dense init: %s\n", err.c_str()); return 1; }
    const Ds41Geometry & G = dense.geom();
    const int n_layer = (int) dense.n_layer(), n_embd = (int) G.n_embd, top_k = (int) G.n_used, nv = (int) G.vocab;
    // enc-skip's layer 20: the LAST kv source (every compressed-KV / index-key write is owned by a kv source, and the
    // layers above it only read that owner's state, so the cut belongs there).  Also the window reach R = swa *
    // (n_layer - cut_layer) is what --enc-replay defaults to.
    int CL = 0;
    for (int64_t s : G.kv_sources) CL = std::max<int>(CL, (int) s);
    // exactness needs every later layer's window reach to stay inside the replay: reach = swa * (n_layer - CL)
    const int64_t reach = (int64_t) G.n_swa * (int64_t) (n_layer - CL);
    const bool enc_bounded = enc_skip && a.enc_bounded && ER < reach;
    if (enc_bounded) {   // DeepInfer's decoder SWA bounded replay: the window + the longest speculative backtrack
        const int64_t least = G.n_swa + 12;
        if (ER < least) { std::fprintf(stderr, "note: --enc-replay %lld < swa + 12; raised to %lld\n", (long long) ER, (long long) least); ER = least; }
        std::fprintf(stderr, "enc-skip: BOUNDED replay of %lld tokens (exact reach %lld): the decoder layers' window is clipped at the"
                     " replay start - approximate, DeepInfer's production scheme\n", (long long) ER, (long long) reach);
    } else if (enc_skip && ER < reach) {
        std::fprintf(stderr, "note: --enc-replay %lld < the window reach %lld (swa %lld x %d layers past %d); raised\n",
                     (long long) ER, (long long) reach, (long long) G.n_swa, n_layer - CL, CL);
        ER = reach;
    }
    if (enc_skip) std::fprintf(stderr, "enc-skip: cut at layer %d (last kv source), replay window %lld (reach %lld)\n", CL, (long long) ER, (long long) reach);
    if (a.engram.empty()) std::fprintf(stderr, "WARNING: no --engram dirs: the Engram add is skipped (not the real model)\n");

    // ---- routed experts
    Ds41FileBlobs src(a.model);
    if (!src.open(err)) { std::fprintf(stderr, "ds41_generate: experts: %s\n", err.c_str()); return 1; }
    const Ds41MoeGeom & TG = src.geom();
    g_routes.init(a.dump_routes, (int) TG.n_layers, (int) TG.n_experts, (int) TG.top_k);
    if (g_routes.on()) std::fprintf(stderr, "routes: counting every (layer, expert) selection -> %s\n", a.dump_routes.c_str());
    if (g_routes.on() && a.serve) {   // the server's counts accumulate across restarts, and are written after each request
        const uint64_t prev = g_routes.load();
        std::fprintf(stderr, "routes: %llu earlier selections loaded from %s\n", (unsigned long long) prev, a.dump_routes.c_str());
    }
    double slot_gib = 0;
    if (a.experts != "cpu" && !ggml_backend_is_cpu(be)) {
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        slot_gib = std::max(0.0, ((double) fr - a.vram_margin_gib * 1073741824.0) / 1073741824.0);
        std::fprintf(stderr, "VRAM: %.2f GiB free after the dense half -> %.2f GiB of expert slots (margin %.2f)\n",
                     fr / 1073741824.0, slot_gib, a.vram_margin_gib);
    }
    Ds41MoeTier tier;
    Ds41MoeConfig mc;
    mc.cpu_only = a.experts == "cpu" || ggml_backend_is_cpu(be);
    mc.peer_devices = a.peer_devices;   // the other cards' expert tiers: none = off, the default path is untouched
    // defer the VRAM cache: the prompt's chunk + graph buffers (~4 GiB) use that VRAM first, then the cache opens into
    // it (measured) - the old fixed --vram-margin had to cover prefill and starved decode of slots.  --serve does this
    // per request: a multi-token prompt closes the cache (release_cache), the decode after it re-seeds (~0.6 s).
    // This is the ds4 engine's `defer_seed`, and for a UNIFORM geometry (DS4/DS4.1: every expert one format, one blob
    // size) it is what makes the reopen happen at all: opened at init the cache holds ~14 GiB, the serve path then
    // measures 0.87 GiB free, releases it - and a uniform geometry has no other seed that opens it again, so the
    // decode ran with no VRAM slots (measured on the Q2 model, 2026-10-17: 6.2 tok/s instead of ~7.9).
    const bool defer_vram = !mc.cpu_only && (a.prefill_chunk > 1 || a.prompt_batch > 1);
    mc.defer_cache = defer_vram;
    mc.slots = a.slots > 0 ? a.slots : TG.n_layers * TG.n_experts;
    mc.slot_gib = a.slots > 0 ? 0.0 : slot_gib;
    mc.pcie_frac = a.pcie;
    mc.tier_batch_pcie = a.tier_batch_pcie;
    mc.pf_b = a.pf_b;   // predicted prefetch (Ds41Dense::predict -> tier.prefetch), experts per layer
    mc.threads = a.threads;
    mc.arena_gib = a.arena_gib;
    mc.max_arena_gib = a.arena_gib;
    mc.vram_lru = a.vram_lru;
    mc.arena_adapt = a.arena_adapt;
    mc.arena_skip_resident = a.arena_skip_vram;   // --arena-skip-vram: no duplicate head in the arena
    mc.chunk_mmq = a.chunk_mmq;
    mc.chunk_prefetch = a.chunk_prefetch;
    mc.chunk_prefetch_gib = a.chunk_prefetch_gib;
    mc.skip_miss = a.skip_miss;
    mc.skip_file = a.skip_file;
    // soft / hard pruning (the GLM recipe, as tools/ds4/ds4_generate): listed experts take no VRAM slot and no arena space;
    // with --prune-penalty X they keep a selection bias -X (still chosen when the router prefers them by more than X, then
    // read from the file tier), without it they are never routed
    std::vector<char> pruned;
    int64_t n_pruned = 0;
    if (!a.prune.empty()) {
        pruned.assign((size_t) (TG.n_layers * TG.n_experts), 0);
        std::FILE * pf = std::fopen(a.prune.c_str(), "r");
        if (!pf) { std::fprintf(stderr, "ds41_generate: cannot open %s\n", a.prune.c_str()); return 2; }
        long l, e;
        while (std::fscanf(pf, "%ld %ld", &l, &e) == 2)
            if (l >= 0 && l < TG.n_layers && e >= 0 && e < TG.n_experts && !pruned[(size_t) (l * TG.n_experts + e)]) {
                pruned[(size_t) (l * TG.n_experts + e)] = 1;
                ++n_pruned;
            }
        std::fclose(pf);
        std::fprintf(stderr, "prune: %lld (layer, expert) pairs %s (%s)\n", (long long) n_pruned,
                     a.prune_penalty > 0 ? ("penalised -" + std::to_string(a.prune_penalty)).c_str() : "masked out of routing", a.prune.c_str());
        mc.exclude = pruned;
    }
    if (!tier.init(TG, &src, mc, err)) { std::fprintf(stderr, "ds41_generate: tier init: %s\n", err.c_str()); return 1; }
    dense.set_oom_hook([&]() { return tier.release_prestage(); });   // a prompt pass that does not fit: prestage gives way
    std::vector<std::pair<int32_t, int32_t>> ranked;
    if (!mc.cpu_only) {
        if (!a.rank.empty()) {
            ranked = ds41_read_ranked(a.rank, err);
            if (ranked.empty()) { std::fprintf(stderr, "ds41_generate: --rank: %s\n", err.c_str()); return 1; }
        } else {
            for (int64_t l = 0; l < TG.n_layers; ++l)
                for (int64_t e = 0; e < TG.n_experts; ++e) ranked.emplace_back((int32_t) l, (int32_t) e);
        }
        const double t0 = now_ms();
        if (defer_vram ? !tier.build_arena_ranked(ranked, err) : !tier.seed_from_ranked(ranked, err)) {
            std::fprintf(stderr, "ds41_generate: seed: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "tier: %lld experts in VRAM, %lld in the RAM arena (%.1f GiB), %lld file tier; seeded in %.1f s\n",
                     (long long) tier.resident(), (long long) tier.arena_experts(), tier.arena_gib(), (long long) tier.file_tier(),
                     (now_ms() - t0) / 1000.0);
    }
    // the selection-only bias per layer: the prune penalty / mask, + route_bias on the experts resident in VRAM now
    std::vector<char> proxy_res;   // --bias-prompt: the would-be VRAM set (first N ranked) until the real seed
    if (a.bias_prompt > 0) {
        proxy_res.assign((size_t) (TG.n_layers * TG.n_experts), 0);
        int got = 0;
        for (const auto & le : ranked) {
            if (got >= a.bias_prompt) break;
            char & r = proxy_res[(size_t) ((int64_t) le.first * TG.n_experts + le.second)];
            if (!r) { r = 1; ++got; }
        }
    }
    auto resident_now = [&](int64_t l, int64_t e) {
        return proxy_res.empty() ? tier.resident(l, e) : proxy_res[(size_t) (l * TG.n_experts + e)] != 0;
    };
    // bias_now: route-bias / file penalty are DECODE levers - before the VRAM seed (the prompt) only the prune part
    // applies, unless --bias-prompt measures them (the prompt's file tier is prefetched; a penalty there buys nothing)
    bool bias_now = a.bias_prompt > 0 || !defer_vram || (std::getenv("STRATA_BIAS_NOW") && std::getenv("STRATA_BIAS_NOW")[0] == '1');
    bool bias_set = false;   // a nonzero bias is in the dense half now (--serve turns it off again for the next prompt)
    auto apply_bias = [&]() {
        const bool want = n_pruned > 0 || (bias_now && (a.route_bias != 0.0f || a.route_bias_file != 0.0f));
        if (!want && !bias_set) return;
        bias_set = want;
        std::vector<float> rb((size_t) TG.n_experts);
        for (int64_t l = 0; l < TG.n_layers; ++l) {
            for (int64_t e = 0; e < TG.n_experts; ++e)
                rb[(size_t) e] = (!pruned.empty() && pruned[(size_t) (l * TG.n_experts + e)]) ? (a.prune_penalty > 0 ? -a.prune_penalty : -1e30f)
                               : (bias_now && a.route_bias != 0.0f && !mc.cpu_only && resident_now(l, e)) ? a.route_bias
                               : (bias_now && a.route_bias_file != 0.0f && !resident_now(l, e) && !tier.in_arena(l, e) && !tier.on_peer(l, e)) ? -a.route_bias_file : 0.0f;
            dense.set_route_bias((int) l, rb.data());
        }
    };
    apply_bias();   // the prune part now (the prompt routes with it); route_bias again once the VRAM cache is seeded
    std::fprintf(stderr, "loaded in %.1f s: %d layers, ctx %lld, prefill chunk %d, experts %s, %zu classes\n",
                 (now_ms() - t_load) / 1000.0, n_layer, (long long) ctx, a.prefill_chunk, tier.mode(), TG.cls_gu.size());

    // ---- the forward pieces
    std::vector<int> ids1((size_t) top_k);
    std::vector<float> w1((size_t) top_k), routed1((size_t) n_embd), logits1((size_t) nv);
    // --serve's short-turn passes (STRATA_SERVE_PASS_M tokens per dense pass; 1 = single decode steps)
    const int pass_max = std::max(1, std::getenv("STRATA_SERVE_PASS_M") ? std::atoi(std::getenv("STRATA_SERVE_PASS_M")) : 16);
    std::vector<float> crouted_m((size_t) std::max(pass_max, 17) * n_embd);   // 17: pass_v's DSpark seeding / verify passes
    std::string ferr;
    // STRATA_DEC_PROF=1: per-stage wall time of the decode step (begin / predict+prefetch / attn_router / tier / finish)
    static const bool dprof = std::getenv("STRATA_DEC_PROF") != nullptr;
    // decode: finish(l) and predict(l+1) as one graph (identical results; STRATA_NO_FUSE_PREDICT=1 = separate graphs)
    static const bool fuse_predict = std::getenv("STRATA_NO_FUSE_PREDICT") == nullptr;
    // STRATA_ADVISE_AHEAD=k (0 = off): at layer l, predict layer l+k's routing from the current state and hint its
    // top STRATA_ADVISE_TOP (8) file-tier experts to the kernel (page-cache readahead; the SSD is idle in decode)
    const int advise_ahead = std::getenv("STRATA_ADVISE_AHEAD") ? std::atoi(std::getenv("STRATA_ADVISE_AHEAD")) : 0;
    const int advise_top = std::getenv("STRATA_ADVISE_TOP") ? std::atoi(std::getenv("STRATA_ADVISE_TOP")) : 8;
    // STRATA_FILE_PRED=1 (measurement only, slower): at layer l, predict layers l+1..l+4 from the current state and count
    // how many of the file-tier misses each layer actually routes were in that prediction's top 4 / 8 / 16 - the hit
    // rate a speculative O_DIRECT read k layers ahead would get.  Printed after the decode line.
    // speculative file-tier reads (decode, OPT-IN): finish(l)'s fused prediction graph also predicts layers
    // l+2..l+1+SPEC_AHEAD; the file-tier experts among each prediction's top STRATA_SPEC_TOP are read ahead
    // (tier.spec_read).  Measured 2026-10-12 (p256 x 256, --slots 1500, tokens identical): 61-81% of the file misses
    // come from the stash and the tier gets ~1.8 ms/token faster, but the bigger finish graph (+1.5 ms) and issuing
    // (+1.3 ms) eat it - net 24.97-25.10 vs 25.56 tok/s off.  Default 0 = off; try STRATA_SPEC_TOP=4.
    const int spec_top = std::getenv("STRATA_SPEC_TOP") ? std::atoi(std::getenv("STRATA_SPEC_TOP")) : 0;
    const int spec_ahead = std::getenv("STRATA_SPEC_AHEAD") ? std::atoi(std::getenv("STRATA_SPEC_AHEAD")) : 2;
    // STRATA_SPEC_L1=1: also read the next layer's predicted file-tier experts (one layer of lead: mostly late)
    const bool spec_l1 = std::getenv("STRATA_SPEC_L1") && std::getenv("STRATA_SPEC_L1")[0] == '1';
    // STRATA_DUMP_MAIN=F (measurement only, slower): per decode step append {int32 token, int32 pos, 3 x n_embd f32} to F
    // = the step's input token and DeepSeek model.py's DSpark main_hidden: the HC-mean of the state ENTERING target
    // layers 37/38/39 ("the MTP head reads the attention input of its target layers, not their output") = the output
    // of layers 36/37/38 - for the offline DSpark acceptance test (tools/v41/dspark_accept.py).  The one-shot run then
    // feeds the last 128 prompt tokens (DSpark's attention window) as steps too.
    const char * dump_main_path = std::getenv("STRATA_DUMP_MAIN");
    std::FILE * dump_main = dump_main_path ? std::fopen(dump_main_path, "wb") : nullptr;
    const int dm_layers[3] = { 36, 37, 38 };   // outputs of these = inputs of the targets 37 / 38 / 39
    std::vector<float> dm_x, dm_pre, dm_row;
    int dm_tok = 0, dm_pos = 0;
    const bool file_pred = std::getenv("STRATA_FILE_PRED") != nullptr;
    constexpr int kFP = 4;
    std::vector<int> fp_ids(file_pred ? (size_t) kFP * (size_t) n_layer * 16 : 0, -1);
    long long fp_miss[kFP + 1] = {0}, fp_hit[kFP + 1][3] = {{0}}, fp_cand[kFP + 1][3] = {{0}};
    // decode: the tier sums the routed experts on the card straight into the dense half's routed tensor (same bits as
    // the host sum; no parts readback / routed upload per layer).  STRATA_DEC_HOSTSUM=1 = the host round trip
    // the routed tensor's device address: re-read after every dense.set_columns (elastic columns reallocate it)
    float * rdev1 = (!mc.cpu_only && !std::getenv("STRATA_DEC_HOSTSUM")) ? dense.routed_dev() : nullptr;
    double dp[6] = {0, 0, 0, 0, 0, 0};
    long dp_n = 0;
    auto step = [&](int tok, int pos) -> bool {   // one token through every layer; logits via dense.logits
        double ta = dprof ? now_ms() : 0;
        if (dump_main) {
            dm_tok = tok;
            dm_pos = pos;
            dm_row.assign((size_t) 3 * n_embd, 0.0f);
        }
        if (!dense.begin(&tok, 1, pos)) { ferr = dense.last_error(); return false; }
        if (dprof) { const double t = now_ms(); dp[0] += t - ta; ta = t; }
        int32_t top16[16];
        bool have_pred = false;
        for (int l = 0; l < n_layer; ++l) {
            if (file_pred) {
                for (int k = 1; k <= kFP; ++k) {
                    int * const pk = fp_ids.data() + ((size_t) (k - 1) * n_layer + (size_t) (l + k < n_layer ? l + k : 0)) * 16;
                    if (l + k < n_layer && !dense.predict(l + k, pk, 16)) std::fill(pk, pk + 16, -1);
                    if (l + k < n_layer)   // file-tier candidates = the reads a speculative fetch would issue
                        for (int i = 0; i < 16; ++i)
                            if (pk[i] >= 0 && !tier.resident(l + k, pk[i]) && !tier.in_arena(l + k, pk[i]))
                                for (int c = 0; c < 3; ++c) fp_cand[k][c] += i < (4 << c);
                }
            }
            if (advise_ahead > 0 && l + advise_ahead < n_layer) {   // file-tier readahead for layer l+k's predicted experts
                int ta[16];
                if (dense.predict(l + advise_ahead, ta, 16)) tier.advise_file(l + advise_ahead, ta, advise_top);
            }
            if (mc.pf_b > 0 && !mc.cpu_only) {   // the DMAs run under this layer's attention
                int t16[16];
                if (have_pred) {                  // predicted by the previous layer's finish_predict (same state)
                    tier.prefetch(l, top16, 16);
                } else if (dense.predict(l, t16, 16)) {
                    for (int i = 0; i < 16; ++i) top16[i] = t16[i];
                    tier.prefetch(l, top16, 16);
                }
                have_pred = false;
            }
            if (dprof) { const double t = now_ms(); dp[1] += t - ta; ta = t; }
            const float * fn = nullptr;
            if (!dense.attn_router(l, ids1.data(), w1.data(), &fn)) { ferr = dense.last_error(); return false; }
            g_routes.add(l, ids1.data(), 1);
            if (dprof) { const double t = now_ms(); dp[2] += t - ta; ta = t; }
            if (file_pred) {
                for (int j = 0; j < top_k; ++j) {
                    const int e = ids1[(size_t) j];
                    if (e < 0 || tier.resident(l, e) || tier.in_arena(l, e)) continue;
                    ++fp_miss[0];
                    for (int k = 1; k <= kFP; ++k) {
                        if (l - k < 0) continue;   // nothing predicted this layer k layers ago
                        const int * pk = fp_ids.data() + ((size_t) (k - 1) * n_layer + (size_t) l) * 16;
                        ++fp_miss[k];
                        int at = 16;
                        for (int i = 0; i < 16; ++i) if (pk[i] == e) { at = i; break; }
                        if (at < 4) ++fp_hit[k][0];
                        if (at < 8) ++fp_hit[k][1];
                        if (at < 16) ++fp_hit[k][2];
                    }
                }
            }
            const bool tok = rdev1 ? tier.run_to_dev(l, ids1.data(), w1.data(), fn, rdev1)
                                   : tier.run(l, ids1.data(), w1.data(), fn, routed1.data());
            if (!tok) { ferr = "tier.run failed"; return false; }
            if (dprof) { const double t = now_ms(); dp[3] += t - ta; ta = t; }
            const float * rsum = rdev1 ? nullptr : routed1.data();   // null: already in the routed tensor
            if (fuse_predict && mc.pf_b > 0 && !mc.cpu_only && l + 1 < n_layer) {
                int t16[16];
                if (spec_top > 0 && spec_ahead > 0) {
                    int ahead[16 * 8];
                    const int na = std::min(spec_ahead, 8);
                    if (!dense.finish_predict_ahead(l, rsum, l + 1, t16, 16, na, ahead)) { ferr = dense.last_error(); return false; }
                    if (dprof) { const double t = now_ms(); dp[4] += t - ta; ta = t; }
                    if (spec_l1) tier.spec_read(l + 1, t16, std::min(spec_top, 16));
                    for (int a = 0; a < na; ++a) tier.spec_read(l + 2 + a, ahead + a * 16, std::min(spec_top, 16));
                    if (dprof) { const double t = now_ms(); dp[5] += t - ta; ta = t; }
                } else if (!dense.finish_predict(l, rsum, l + 1, t16, 16)) { ferr = dense.last_error(); return false; }
                for (int i = 0; i < 16; ++i) top16[i] = t16[i];
                have_pred = true;
            } else if (!dense.finish(l, rsum)) { ferr = dense.last_error(); return false; }
            if (dprof) { const double t = now_ms(); dp[4] += t - ta; ta = t; }
            for (int j = 0; dump_main && j < 3; ++j) {
                if (l != dm_layers[j]) continue;
                dense.read_state(dm_x, dm_pre);   // [n_embd, HC] for the one token
                const size_t HC = dm_x.size() / (size_t) n_embd;
                for (size_t h = 0; h < HC; ++h)
                    for (int64_t d = 0; d < n_embd; ++d) dm_row[(size_t) j * n_embd + d] += dm_x[h * n_embd + d] / (float) HC;
                if (j == 2) {
                    std::fwrite(&dm_tok, 4, 1, dump_main);
                    std::fwrite(&dm_pos, 4, 1, dump_main);
                    std::fwrite(dm_row.data(), 4, dm_row.size(), dump_main);
                }
            }
        }
        if (dprof && ++dp_n % 64 == 0)
            std::fprintf(stderr, "decode step ms/token: begin %.2f predict %.2f attn_router %.2f tier %.2f finish %.2f spec %.2f (%ld steps)\n",
                         dp[0] / dp_n, dp[1] / dp_n, dp[2] / dp_n, dp[3] / dp_n, dp[4] / dp_n, dp[5] / dp_n, dp_n);
        return true;
    };
    // a few tokens in ONE dense pass (the dense weights - ~8 GB - read once for m tokens, not per token), the routed
    // experts token by token from the VRAM cache (run_to_dev into the pass's routed rows): --serve's short turns.
    // Row-batched matmuls: the same model, not the same bits as m single steps.
    std::vector<int> mids;
    std::vector<float> mw;
    auto pass_m = [&](const int * toks, int m, int pos) -> bool {
        mids.resize((size_t) m * top_k);
        mw.resize((size_t) m * top_k);
        if (!dense.begin(toks, m, pos)) { ferr = dense.last_error(); return false; }
        for (int l = 0; l < n_layer; ++l) {
            const float * fn = nullptr;
            if (!dense.attn_router(l, mids.data(), mw.data(), &fn)) { ferr = dense.last_error(); return false; }
            g_routes.add(l, mids.data(), m);
            // STRATA_SERVE_PASS_BATCH=1 (opt-in): the pass's tokens go to the tier as the verify pass's do - one
            // run_multi_to_dev per kMaxMulti tokens, the union of their experts served once (a miss is read / DMA'd /
            // computed once for every token routing it) - instead of one run_to_dev per token
            static const bool pass_batch = std::getenv("STRATA_SERVE_PASS_BATCH") && std::getenv("STRATA_SERVE_PASS_BATCH")[0] == '1';
            if (pass_batch && rdev1) {
                constexpr int kMM = Ds41MoeTier::kMaxMulti;
                for (int off = 0; off < m; off += kMM) {
                    const int nn = std::min(kMM, m - off);
                    if (!tier.run_multi_to_dev(l, nn, mids.data() + (size_t) off * top_k, mw.data() + (size_t) off * top_k,
                                               fn + (size_t) off * n_embd, rdev1 + (size_t) off * n_embd)) {
                        ferr = "tier.run_multi_to_dev failed"; return false;
                    }
                }
                if (!dense.finish(l, nullptr)) { ferr = dense.last_error(); return false; }
                continue;
            }
            for (int t = 0; t < m; ++t) {
                const bool ok = rdev1 ? tier.run_to_dev(l, mids.data() + (size_t) t * top_k, mw.data() + (size_t) t * top_k,
                                                        fn + (size_t) t * n_embd, rdev1 + (size_t) t * n_embd)
                                      : tier.run(l, mids.data() + (size_t) t * top_k, mw.data() + (size_t) t * top_k,
                                                 fn + (size_t) t * n_embd, routed1.data());
                if (!ok) { ferr = "tier.run failed"; return false; }
                if (!rdev1) std::copy(routed1.begin(), routed1.end(), crouted_m.begin() + (size_t) t * n_embd);
            }
            if (!dense.finish(l, rdev1 ? nullptr : crouted_m.data())) { ferr = dense.last_error(); return false; }
        }
        return true;
    };
    // pass_m + DSpark's main_hidden: vh[t * 3D + j * D ..] = the HC-mean of the state leaving layer dm_layers[j] (=
    // entering the DSpark target layer 37 + j) for pass token t (the STRATA_DUMP_MAIN math, token-major read_state);
    // vlog = the pass's m rows of logits.
    std::vector<float> vh, vlog, vx, vpre;
    std::vector<int> pv_pred, pv_rank;                 // D3 prefetch: per-token predictions, the ranked union
    // STRATA_D3_FILE_PRED=1 (measurement only, slower): at layer l of a pass, predict_multi(l+1) and (l+2); at layer l
    // count, for every routed expert that is on NO tier but the file, whether the prediction made 1 / 2 layers earlier
    // had it in the row's top 2 / 4 / 8 / 16 (row) or in any row's top-m (union), and how many file-tier experts those
    // predictions would have read (issued) - recall and precision of a speculative read k layers ahead.
    static const bool d3_fpred = std::getenv("STRATA_D3_FILE_PRED") != nullptr;
    std::vector<int> fpv[2];            // [k-1]: n_layer x kMaxMulti x 16 predicted ids for each layer
    std::vector<char> fpv_ok[2];        // [k-1][layer]: predicted in this pass
    static long qfp_file = 0, qfp_hit[2][4] = {}, qfp_uhit[2][4] = {}, qfp_iss[2][4] = {}, qfp_calls = 0, qfp_pass = 0;
    std::vector<std::pair<int, int>> pv_score;
    double pv_ms[4] = {0, 0, 0, 0};   // pass_v split: dense (begin/attn_router/finish), tier, taps (read_state), logits
    double pvd_ms[4] = {0, 0, 0, 0};  // the dense bucket's own split: begin, predict (+ prefetch issue), attn_router, finish
    long pv_n = 0, pv_tok = 0;
    // --dense-bench (and STRATA_TIER_SKIP=1, measurement only): the expert tier is not run - the routed sums stay
    // whatever the routed tensor holds (the bench zeroes it once), and no prefetch is issued.  Garbage text.
    bool tier_skip = std::getenv("STRATA_TIER_SKIP") && std::getenv("STRATA_TIER_SKIP")[0] == '1';
    auto pass_v = [&](const int * toks, int m, int pos, const Ds41Dense::Row * rows = nullptr) -> bool {
        // D3 (the batched per-class verify tier) is the default: correct (bias-off verify-vs-prefill gate, near-ties
        // only) and faster (verify 167.6 -> ~100 ms/pass).  STRATA_TIER_BATCH=0 = the per-token run_to_dev path.
        static const bool no_tier_batch = [] { const char* v = std::getenv("STRATA_TIER_BATCH"); return v && v[0] == '0'; }();
        double tq = now_ms();
        auto lap = [&](int i) { const double t = now_ms(); pv_ms[i] += t - tq; tq = t; };
        double tqd = tq;
        auto lapd = [&](int i) { const double t = now_ms(); pvd_ms[i] += t - tqd; tqd = t; };
        ++pv_n; pv_tok += m;
        mids.resize((size_t) m * top_k);
        mw.resize((size_t) m * top_k);
        vh.assign((size_t) m * 3 * n_embd, 0.0f);
        // a batch window (rows): rows of several conversation slots in one pass (Ds41Dense::begin_rows), else this conversation's
        if (rows ? !dense.begin_rows(rows, m) : !dense.begin(toks, m, pos)) { ferr = dense.last_error(); return false; }
        lapd(0);
        if (d3_fpred) {
            for (int k = 0; k < 2; ++k) {
                fpv[k].assign((size_t) n_layer * Ds41MoeTier::kMaxMulti * 16, -1);
                fpv_ok[k].assign((size_t) n_layer, 0);
            }
            ++qfp_pass;
        }
        // D3 prefetch (STRATA_D3_PF = slots per layer, default 1, 0 = off): before each layer's attention, predict
        // every pass token's routing (predict_multi), rank the union (score 16 - rank, summed over tokens) and DMA the
        // likely misses into the prefetch slots on the prefetch stream - they land under the attention.  PCIe is the
        // wall (nsys: ~100 expert DMAs per verify pass, 1.4 s of a 4.2 s decode), so a wrong guess costs a full blob:
        // docs prompt, same text: slots 0/1/2/3/6 -> verify 104.4/101.7/104.5/111.1/124.1 ms (precision 40% at 1).
        // Its row cap stays 8 (the old kMaxMulti, i.e. the passes it was measured on): since kMaxMulti 16 a 9-16-row
        // pass is ONE tier call too, but it ran without the prefetch before and still does (a separate experiment).
        static const int d3_pf = std::getenv("STRATA_D3_PF") ? std::atoi(std::getenv("STRATA_D3_PF")) : 1;
        constexpr int kD3PfRows = 8;
        static_assert(kD3PfRows <= Ds41MoeTier::kMaxMulti, "the prefetch is consumed by ONE tier call per layer");
        const bool pf_now = rdev1 && !no_tier_batch && d3_pf > 0 && m <= kD3PfRows && !tier_skip;
        bool have_pred = false;   // this layer's prediction came with the previous layer's finish (finish_predict_multi)
        for (int l = 0; l < n_layer; ++l) {
            if (pf_now) {
                pv_pred.resize((size_t) m * 16);
                if (have_pred || dense.predict_multi(l, pv_pred.data())) {
                    pv_score.clear();
                    for (int t = 0; t < m; ++t)
                        for (int i = 0; i < 16; ++i) {
                            const int e = pv_pred[(size_t) t * 16 + i];
                            if (e < 0) continue;
                            auto it = std::find_if(pv_score.begin(), pv_score.end(), [&](const std::pair<int, int> & p) { return p.first == e; });
                            if (it == pv_score.end()) pv_score.push_back({ e, 16 - i }); else it->second += 16 - i;
                        }
                    std::stable_sort(pv_score.begin(), pv_score.end(), [](const std::pair<int, int> & a, const std::pair<int, int> & b) { return a.second > b.second; });
                    pv_rank.clear();
                    for (const auto & p : pv_score) pv_rank.push_back(p.first);
                    tier.prefetch_multi(l, pv_rank.data(), (int) pv_rank.size(), d3_pf);
                }
            }
            // D3 speculative NVMe reads (STRATA_D3_SPEC=n, default 0 = off; STRATA_D3_SPEC_AHEAD=k, default 2): the
            // pass's routing for layer l+k predicted from the state entering l (predict_multi, bias included), its
            // union ranked as above, and the NVMe-tier experts among the top n read into the stash by background
            // readers while layers l..l+k-1 run - the verify pass otherwise waits on them (nsys x0.5: pthread_join
            // 7.9 ms/token on the decode thread).  A wrong guess costs SSD bandwidth only.
            static const int d3_spec = std::getenv("STRATA_D3_SPEC") ? std::atoi(std::getenv("STRATA_D3_SPEC")) : 0;
            static const int d3_spec_ahead = std::getenv("STRATA_D3_SPEC_AHEAD") ? std::atoi(std::getenv("STRATA_D3_SPEC_AHEAD")) : 2;
            if (rdev1 && !no_tier_batch && d3_spec > 0 && m <= Ds41MoeTier::kMaxMulti && l + d3_spec_ahead < n_layer) {
                pv_pred.resize((size_t) m * 16);
                if (dense.predict_multi(l + d3_spec_ahead, pv_pred.data())) {
                    pv_score.clear();
                    for (int t = 0; t < m; ++t)
                        for (int i = 0; i < 16; ++i) {
                            const int e = pv_pred[(size_t) t * 16 + i];
                            if (e < 0) continue;
                            auto it = std::find_if(pv_score.begin(), pv_score.end(), [&](const std::pair<int, int> & q) { return q.first == e; });
                            if (it == pv_score.end()) pv_score.push_back({ e, 16 - i }); else it->second += 16 - i;
                        }
                    std::stable_sort(pv_score.begin(), pv_score.end(), [](const std::pair<int, int> & a, const std::pair<int, int> & b) { return a.second > b.second; });
                    pv_rank.clear();
                    for (const auto & q : pv_score) pv_rank.push_back(q.first);
                    tier.spec_read(l + d3_spec_ahead, pv_rank.data(), std::min(d3_spec, (int) pv_rank.size()), true);
                }
            }
            if (d3_fpred && m <= Ds41MoeTier::kMaxMulti) {
                for (int k = 1; k <= 2; ++k) {
                    if (l + k >= n_layer) continue;
                    int * dst = fpv[k - 1].data() + (size_t) (l + k) * Ds41MoeTier::kMaxMulti * 16;
                    if (!dense.predict_multi(l + k, dst)) continue;
                    fpv_ok[k - 1][(size_t) (l + k)] = 1;
                    for (int c = 0; c < 4; ++c) {   // issued: distinct file-tier experts in the union of the rows' top-m
                        const int mtop = 2 << c;
                        std::vector<int> seen;
                        for (int t = 0; t < m; ++t)
                            for (int i = 0; i < mtop; ++i) {
                                const int e = dst[(size_t) t * 16 + i];
                                if (e < 0 || std::find(seen.begin(), seen.end(), e) != seen.end()) continue;
                                seen.push_back(e);
                                if (!tier.resident(l + k, e) && !tier.in_arena(l + k, e) && !tier.on_peer(l + k, e)) ++qfp_iss[k - 1][c];
                            }
                    }
                }
            }
            const float * fn = nullptr;
            lapd(1);
            if (!dense.attn_router(l, mids.data(), mw.data(), &fn)) { ferr = dense.last_error(); return false; }
            lapd(2);
            g_routes.add(l, mids.data(), m);
            if (d3_fpred && m <= Ds41MoeTier::kMaxMulti) {
                ++qfp_calls;
                std::vector<int> seen;
                for (int t = 0; t < m; ++t)
                    for (int j = 0; j < top_k; ++j) {
                        const int e = mids[(size_t) t * top_k + j];
                        if (e < 0 || std::find(seen.begin(), seen.end(), e) != seen.end()) continue;
                        seen.push_back(e);
                        if (tier.resident(l, e) || tier.in_arena(l, e) || tier.on_peer(l, e)) continue;
                        ++qfp_file;
                        for (int k = 1; k <= 2; ++k) {
                            if (!fpv_ok[k - 1][(size_t) l]) continue;
                            const int * pr = fpv[k - 1].data() + (size_t) l * Ds41MoeTier::kMaxMulti * 16;
                            int best_row = 16, best_any = 16;   // the expert's best rank in its own rows / in any row
                            for (int t2 = 0; t2 < m; ++t2) {
                                bool routes = false;
                                for (int j2 = 0; j2 < top_k && !routes; ++j2) routes = mids[(size_t) t2 * top_k + j2] == e;
                                for (int i = 0; i < 16; ++i)
                                    if (pr[(size_t) t2 * 16 + i] == e) { best_any = std::min(best_any, i); if (routes) best_row = std::min(best_row, i); break; }
                            }
                            for (int c = 0; c < 4; ++c) { if (best_row < (2 << c)) ++qfp_hit[k - 1][c]; if (best_any < (2 << c)) ++qfp_uhit[k - 1][c]; }
                        }
                    }
                if (qfp_calls % 1280 == 0)
                    std::fprintf(stderr, "[fpred] %ld calls (%ld passes): %.2f file experts/call; recall row-top2/4/8/16 k1 %.2f/%.2f/%.2f/%.2f k2 %.2f/%.2f/%.2f/%.2f; "
                                 "union k1 %.2f/%.2f/%.2f/%.2f k2 %.2f/%.2f/%.2f/%.2f; issued/call k1 %.2f/%.2f/%.2f/%.2f k2 %.2f/%.2f/%.2f/%.2f\n",
                                 qfp_calls, qfp_pass, (double) qfp_file / qfp_calls,
                                 (double) qfp_hit[0][0] / std::max(1L, qfp_file), (double) qfp_hit[0][1] / std::max(1L, qfp_file), (double) qfp_hit[0][2] / std::max(1L, qfp_file), (double) qfp_hit[0][3] / std::max(1L, qfp_file),
                                 (double) qfp_hit[1][0] / std::max(1L, qfp_file), (double) qfp_hit[1][1] / std::max(1L, qfp_file), (double) qfp_hit[1][2] / std::max(1L, qfp_file), (double) qfp_hit[1][3] / std::max(1L, qfp_file),
                                 (double) qfp_uhit[0][0] / std::max(1L, qfp_file), (double) qfp_uhit[0][1] / std::max(1L, qfp_file), (double) qfp_uhit[0][2] / std::max(1L, qfp_file), (double) qfp_uhit[0][3] / std::max(1L, qfp_file),
                                 (double) qfp_uhit[1][0] / std::max(1L, qfp_file), (double) qfp_uhit[1][1] / std::max(1L, qfp_file), (double) qfp_uhit[1][2] / std::max(1L, qfp_file), (double) qfp_uhit[1][3] / std::max(1L, qfp_file),
                                 (double) qfp_iss[0][0] / qfp_calls, (double) qfp_iss[0][1] / qfp_calls, (double) qfp_iss[0][2] / qfp_calls, (double) qfp_iss[0][3] / qfp_calls,
                                 (double) qfp_iss[1][0] / qfp_calls, (double) qfp_iss[1][1] / qfp_calls, (double) qfp_iss[1][2] / qfp_calls, (double) qfp_iss[1][3] / qfp_calls);
            }
            lap(0);
            if (tier_skip) {
                // measurement only: no experts (see tier_skip above)
            } else if (rdev1 && !no_tier_batch) {
                // D3: ONE batched per-class tier call for the pass's m tokens (the union of their experts is served
                // once), not m run_to_dev calls.  Split at Ds41MoeTier::kMaxMulti (16: a depth-4 pass and a 16-row
                // lookup verify / prompt pass are one call each).
                constexpr int kMM = Ds41MoeTier::kMaxMulti;
                for (int off = 0; off < m; off += kMM) {
                    const int chunk = std::min(kMM, m - off);
                    if (!tier.run_multi_to_dev(l, chunk, mids.data() + (size_t) off * top_k, mw.data() + (size_t) off * top_k,
                                               fn + (size_t) off * n_embd, rdev1 + (size_t) off * n_embd)) {
                        ferr = "tier.run_multi_to_dev failed"; return false;
                    }
                }
            } else {
                for (int t = 0; t < m; ++t) {
                    const bool ok = rdev1 ? tier.run_to_dev(l, mids.data() + (size_t) t * top_k, mw.data() + (size_t) t * top_k,
                                                            fn + (size_t) t * n_embd, rdev1 + (size_t) t * n_embd)
                                          : tier.run(l, mids.data() + (size_t) t * top_k, mw.data() + (size_t) t * top_k,
                                                     fn + (size_t) t * n_embd, routed1.data());
                    if (!ok) { ferr = "tier.run failed"; return false; }
                    if (!rdev1) std::copy(routed1.begin(), routed1.end(), crouted_m.begin() + (size_t) t * n_embd);
                }
            }
            lap(1);
            tqd = tq;
            // finish + the next layer's prediction in one graph when the prefetch wants the prediction (FZ_FPRED)
            have_pred = false;
            if (pf_now && rdev1 && l + 1 < n_layer && dense.finish_predict_multi(l, l + 1, pv_pred.data())) {
                have_pred = true;
            } else if (!dense.finish(l, rdev1 ? nullptr : crouted_m.data())) { ferr = dense.last_error(); return false; }
            lap(0);
            lapd(3);
            for (int j = 0; j < 3; ++j) {
                if (l != dm_layers[j]) continue;
                if (!dense.read_state(vx, vpre)) { ferr = dense.last_error(); return false; }
                const size_t HC = vx.size() / ((size_t) m * n_embd);
                for (int t = 0; t < m; ++t)
                    for (size_t h = 0; h < HC; ++h) {
                        const float * xs = vx.data() + ((size_t) t * HC + h) * n_embd;
                        float * dst = vh.data() + (size_t) t * 3 * n_embd + (size_t) j * n_embd;
                        for (int64_t d = 0; d < n_embd; ++d) dst[d] += xs[d] / (float) HC;
                    }
                lap(2);
            }
        }
        vlog.resize((size_t) m * nv);
        if (!dense.logits(0, m, vlog.data())) { ferr = dense.last_error(); return false; }
        lap(3);
        return true;
    };
    const int NP = std::max(1, a.prefill_chunk);
    const int SB = std::max(1, S);                 // the dense sub-batch size
    std::vector<int> cids((size_t) NP * top_k);
    std::vector<float> cw((size_t) NP * top_k), crouted((size_t) NP * n_embd), clog;
    const bool routed_host = std::getenv("STRATA_ROUTED_HOST") != nullptr;
    std::vector<float> brouted;                    // the layer-major path's routed sums (n_embd * B)
    if (PB > 0) brouted.resize((size_t) n_embd * (size_t) PB);
    // a single run_chunk call streams layer l's experts once for its n tokens; the card's chunk buffers (dominated by
    // c_parts = n * top_k * n_embd * 4) must fit ~1 GiB, so a larger batch is run in pieces (each streams once again)
    // budget 3 GB by default (STRATA_CHUNK_BUDGET_GB): at B=8192 the old 1 GB cap split every layer into 2 pieces and each
    // piece re-uploaded the whole layer over PCIe (1.97M vs 0.98M expert copies); the prompt has ~12 GiB of VRAM free
    // (the VRAM expert cache is only seeded after the prompt, from the memory measured then)
    const double chunk_budget = std::getenv("STRATA_CHUNK_BUDGET_GB") ? std::atof(std::getenv("STRATA_CHUNK_BUDGET_GB")) * 1e9 : 3.0e9;
    const int piece = (int) std::max<int64_t>(1, (int64_t) chunk_budget / std::max<int64_t>(1, (int64_t) n_embd * 4 * (top_k + 3)));
    // the prompt ids[0, n) at positions pos0..: last_logits = the last row; all_logits (if given) every row
    std::vector<float> last_logits((size_t) nv);
    // the deferred VRAM seed after a prompt: the prompt's buffers go back, the cache opens into the free space, and the
    // decode biases (route bias + file penalty) switch on.  Returns the free GiB measured (for the log).
    double seed_free = 0, seed_gib = 0, seed_s = 0;
    // STRATA_DSPARK=<sidecar> (one-shot, greedy): DSpark speculative decoding - the checkpoint's mtp.0-2 draft stack
    // (ds41_dspark) drafts STRATA_DSPARK_DEPTH (default 4, <= 5) tokens, one dense pass verifies them (pass_v).  The
    // drafter loads here, before the VRAM expert cache measures the free memory, so the cache sizes around it.
    // --serve: STRATA_SERVE_DSPARK=<sidecar> (opt-in) - speculative SAMPLING (spec_accept), same output distribution
    const char * dsp_path = a.serve ? std::getenv("STRATA_SERVE_DSPARK") : std::getenv("STRATA_DSPARK");
    if (a.serve && dsp_path && (!defer_vram || S < 17)) {
        std::fprintf(stderr, "serve: STRATA_SERVE_DSPARK needs the GPU layer-major prompt path - DSpark off\n");
        dsp_path = nullptr;
    }
#if defined(DS41_HAVE_DSPARK)
    std::unique_ptr<Ds41DSpark> dsp;
    ggml_backend_t dsp_backend = nullptr;   // STRATA_DSPARK_DEVICE: the drafter's own backend on another card
#endif
    // --serve: the VRAM expert cache's size for the server's Monitor tab ("Experts in VRAM"), each time it opens or
    // shuts (the deferred seed opens it only after the first prompt; a layer-major batch shuts it for the prompt)
    auto info_vram = [&]() {
        std::printf("INFO expert_slots=%lld expert_cache_mib=%lld\n", (long long) tier.vram_experts(),
                    (long long) (tier.vram_bytes() >> 20));
        std::fflush(stdout);
    };
    auto open_cache = [&]() -> bool {
        dense.release_pass();
#if defined(DS41_HAVE_DSPARK)
        if (dsp_path && !dsp) {
            Ds41DSparkConfig sc;
            sc.backend = be;
            sc.sidecar = dsp_path;
            sc.main_gguf = a.model;
            sc.max_append = 16;
            // STRATA_DSPARK_DEVICE=N (the N-th GPU in HIP_VISIBLE_DEVICES order): the drafter on another card - its
            // 4.9 GB of weights (+ its own copy of the head, 0.7 GB) leave the primary ~270 expert slots more; the
            // main_hidden rows and the drafts cross through host memory either way.  Shrink that card's
            // STRATA_PEER_GIB by ~6 GiB.
            static const int dsp_dev = std::getenv("STRATA_DSPARK_DEVICE") ? std::atoi(std::getenv("STRATA_DSPARK_DEVICE")) : -1;
            if (dsp_dev >= 0) {
                int gpu_i = 0;
                ggml_backend_dev_t pick = nullptr;
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
                    if (gpu_i++ == dsp_dev) { pick = d; break; }
                }
                if (!pick) { err = "STRATA_DSPARK_DEVICE: no such GPU"; return false; }
                if (!dsp_backend) dsp_backend = ggml_backend_dev_init(pick, nullptr);
                if (!dsp_backend) { err = "STRATA_DSPARK_DEVICE: backend init failed"; return false; }
                sc.backend = dsp_backend;
                std::fprintf(stderr, "\nDSpark: on %s (STRATA_DSPARK_DEVICE=%d, its own head)", ggml_backend_dev_name(pick), dsp_dev);
            }
            if (!std::getenv("STRATA_DSPARK_OWN_HEAD") && dsp_dev < 0) sc.shared_head = dense.weight("output.weight");   // -0.7 GB
            dsp = std::make_unique<Ds41DSpark>();
            const double ti = now_ms();
            if (!dsp->init(sc, err)) { err = "DSpark init: " + err; return false; }
            std::fprintf(stderr, "\nDSpark: %s, %.2f GB of weights on the card (%.1f s)", dsp_path, dsp->weight_bytes() / 1e9,
                         (now_ms() - ti) / 1000.0);
        }
#endif
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        seed_free = fr / 1073741824.0;
        seed_gib = std::max(0.0, ((double) fr - a.vram_margin_gib * 1073741824.0) / 1073741824.0);
        // A slot budget of 0 is the tier's "NO LIMIT" (--slots given: the ranked head, up to `slots` experts).  The
        // AUTO path passed it too whenever the card had no room (`free <= margin` clamps seed_gib to 0) - and "no
        // limit" there asked for the WHOLE model's experts: s13 run8 hit exactly that (142.38 GiB requested, 0.53
        // GiB free -> "VRAM seed failed" -> HTTP 400).  With no usable room the cache stays SHUT for this seed and
        // the arena + file tier serve every miss: a slower decode, never a failed request.
        const double min_gib = 1.0;   // below this the admission traffic is not worth the slots
        const bool explicit_slots = a.slots > 0;
        const bool cache_room = explicit_slots || seed_gib >= min_gib;
        tier.set_cache_enabled(cache_room);
        tier.set_slot_gib(explicit_slots ? 0.0 : seed_gib);
        if (!cache_room) std::fprintf(stderr, "serve: VRAM %.2f GiB free (margin %.2f) - no room for the expert cache, "
                                             "the arena + file tier serve the decode\n", seed_free, a.vram_margin_gib);
        const double ts = now_ms();
        if (!tier.seed_from_ranked(ranked, err)) {
            // the cache's own allocation is what fails on a busy card (the arena is host RAM and is built first):
            // one more try with the cache off, so the request still runs
            if (cache_room) {
                std::fprintf(stderr, "serve: cache seed failed (%s) - seeding again with the VRAM cache off\n", err.c_str());
                tier.set_cache_enabled(false);
                if (!tier.seed_from_ranked(ranked, err)) return false;
            } else return false;
        }
        seed_s = (now_ms() - ts) / 1000.0;
        proxy_res.clear();   // the real VRAM set from here on
        bias_now = true;     // decode: route bias + file penalty on
        apply_bias();
        if (a.serve) info_vram();
        return true;
    };
    double fd_dense = 0, fd_tier = 0, fd_logits = 0;   // prompt wall split (printed with the prompt line)
    bool enc_reported = false;
    // ---- env-gated per-layer dumps (determinism bisect): STRATA_LAYER_DUMP=<dir> writes, for every layer of the
    // sequential chunk path, the routed experts' input (fn, the dense half's output) and their weighted sum, plus an
    // index source layer's top-k ids.  Two runs' files diff at the first layer where a run-to-run difference appears,
    // and fn-vs-routed tells whether it is the MoE tier or the attention / indexer half.
    const char * dump_dir = std::getenv("STRATA_LAYER_DUMP");
    auto dumpf = [&](const char * name, const void * p, size_t bytes) {
        std::ofstream f(std::string(dump_dir) + "/" + name, std::ios::binary);
        f.write((const char *) p, (std::streamsize) bytes);
    };
    // tail capture (STRATA_SERVE_DUMP_TAIL, the quality scorer): the last `tail_keep` rows of a feed's logits into
    // tail_logits (row n - tail_keep first) - a teacher-forced continuation's rows without the whole prompt's (a 23K-token
    // prompt's full logits would be 12 GB).  0 = off.
    int tail_keep = 0;
    std::vector<float> tail_logits;
    auto capture_tail = [&](int n, int row0, int nrows, int dense_row0) -> bool {   // rows [row0, row0+nrows) of this feed
        if (tail_keep <= 0) return true;
        const int t0 = n - tail_keep, lo = std::max(row0, t0), hi = row0 + nrows;
        if (lo >= hi) return true;
        if (tail_logits.size() != (size_t) tail_keep * (size_t) nv) tail_logits.assign((size_t) tail_keep * (size_t) nv, 0.0f);
        if (!dense.logits(dense_row0 + (lo - row0), hi - lo, tail_logits.data() + (size_t) (lo - t0) * nv)) { ferr = dense.last_error(); return false; }
        return true;
    };
    int bcap_n = 0, bcap_got = 0;  // serve DSpark seed captured from a layer-major batch (see the layer loop below)
    std::vector<float> bcap_vh;    // [bcap_n][3 * n_embd], as pass_v's vh
    // the layer-major batch width for THIS turn: PB, clamped per request by the VRAM actually free (the --serve loop
    // sets it; a one-shot run keeps PB)
    int pb_cap = PB;
    // 2026-10-23 (after Project Maya v1.1.20 "Stop stops, even while a long prompt is read"): --serve sets feed_cancel
    // to a check of the command queue; a read that is cancelled ends after the layer it is in (a layer-major batch) or
    // the pass it is in, and feed_done says how many of the feed's tokens were read COMPLETELY - the batch being read is
    // abandoned (its rings restored to the batch's start; the rows it wrote lie past that point and are overwritten
    // later), so the conversation stays loaded up to feed_done and the next request continues from there.
    std::function<bool()> feed_cancel;
    int feed_done = 0;
    bool feed_cancelled = false;
    // STRATA_VRAM_TRACE=1: every allocation-relevant step prints the free VRAM (debugging a shared card)
    const bool vram_trace = std::getenv("STRATA_VRAM_TRACE") != nullptr;
    auto vram_log = [&](const char * where, int a) {
        if (!vram_trace) return;
        size_t tf = 0, tt = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &tf, &tt);
        std::fprintf(stderr, "vram trace: %s %d free %.2f GiB\n", where, a, tf / 1073741824.0);
    };
    auto feed = [&](const int * ids, int n, int pos0, const std::function<void(int)> & progress,
                    std::vector<float> * all_logits) -> bool {
        feed_done = 0;
        feed_cancelled = false;
        auto give_up = [&](int done, bool restore_rings) -> bool {   // a cancel: leave feed() as its normal end does
            if (restore_rings) dense.restore();
            feed_done = done;
            feed_cancelled = true;
            ferr = "cancelled";
            if (dc.elastic_cols && dense.set_columns(0) && rdev1) rdev1 = dense.routed_dev();
            tier.release_chunk();
            return false;
        };
        for (int c0 = 0; c0 < n; ) {
            if (PB > 1 && n - c0 >= 2) {
                // ---- layer-major: up to B tokens of every layer, the experts run once per layer over the whole batch
                int B = std::min(pb_cap, n - c0);
                // never leave ONE column for the end: it takes the sequential chunk path below, which does not capture the
                // DSpark seed rows, so a prompt of 8192 k + 1 tokens failed with "DSpark batch seed captured nothing"
                // (found 2026-10-22 with a 16385-token prompt; ~1 in 8192 big reads)
                if (B > 2 && n - c0 - B == 1) --B;
                // The batch must FIT: its state columns (~0.31 MiB a column) and its graph (~0.17 MiB a column,
                // 8192 -> 1424 MiB) come out of whatever is left after everything else sharing the card (a
                // desktop's STT server, DSpark's 2.9 GB, the dense half).  A batch that does not fit is retried at
                // HALF the width (twice the sweeps over the file tier, the same answer) down to SB, so a tight card
                // makes a prompt slower instead of failing the request (s13 run8: "prefill failed: graph allocation
                // failed" on a 21K prompt with Whisper's 2.5 GB resident).
                for (;;) {
                    if (B > dense.columns()) {
                        if (dense.set_columns(B)) { if (rdev1) rdev1 = dense.routed_dev(); }
                        else if (B > SB) {
                            std::fprintf(stderr, "ds41_generate: %d state columns did not fit (%s) - retrying at %d\n",
                                         B, dense.last_error().c_str(), std::max(SB, B / 2));
                            B = std::max(SB, B / 2);
                            continue;
                        } else { ferr = dense.last_error(); return false; }
                    }
                    if (enc_bounded) {   // the replay start of THIS batch (the cut below, rounded to a sub-batch): the decoder masks
                        const int64_t gpos = (int64_t) pos0 + c0, ta = (int64_t) pos0 + n - ER - gpos;
                        int cb = ta <= 0 ? 0 : ta >= B ? B : (int) ta;
                        cb = (cb / SB) * SB;
                        dense.set_bounded_replay(CL, cb > 0 ? gpos + cb : -1);
                    }
                    if (dense.begin_batch(ids + c0, B, pos0 + c0)) break;
                    if (B > SB) {
                        std::fprintf(stderr, "ds41_generate: batch of %d did not fit (%s) - retrying at %d\n",
                                     B, dense.last_error().c_str(), std::max(SB, B / 2));
                        B = std::max(SB, B / 2);
                        continue;
                    }
                    ferr = dense.last_error();
                    return false;
                }
                // enc-skip: columns whose global position is before `pos0 + n - ER` are phase A (layers past layer 20
                // are never read for them), rounded DOWN to a sub-batch boundary so a straddling sub-batch still runs
                // the full path (more work, never wrong); the last sub-batches holding the final token are always full.
                int cut = 0;
                if (enc_skip) {
                    const int64_t gpos = (int64_t) pos0 + c0;
                    const int64_t ta = (int64_t) pos0 + n - ER - gpos;
                    cut = ta <= 0 ? 0 : ta >= B ? B : (int) ta;
                    cut = (cut / SB) * SB;
                    if (!enc_reported) { enc_reported = true; std::fprintf(stderr, "enc-skip: first batch B=%d, cut=%d (phase A %d cols, phase B %d)\n", B, cut, cut, B - cut); }
                }
                if (feed_cancel) dense.snapshot(pos0 + c0);   // the rings at this batch's start (a cancel inside it restores them)
                double t0 = now_ms();
                for (int l = 0; l < n_layer; ++l) {
                    if (feed_cancel && l > 0 && feed_cancel()) return give_up(c0, true);
                    if (l < 4) vram_log("batch layer", l);
                    // layer 20 is the last kv source: for phase A only its persistent KV writes run; layers 21-39 run
                    // nothing at all for phase A (their output columns are never read, and no later layer reads state
                    // a phase-A position would leave past layer 20)
                    const int d0 = (cut > 0 && l >= CL) ? cut : 0;
                    if (cut > 0 && l == CL) {
                        for (int s = 0; s < cut; s += SB) {
                            const int sn = std::min(SB, cut - s);
                            if (!dense.attn_router_span_kv(l, s, sn)) { ferr = dense.last_error(); return false; }
                        }
                    }
                    for (int s = d0; s < B; s += SB) {   // sub-batches in order: the layer's KV state advances as today
                        const int sn = std::min(SB, B - s);
                        if (!dense.attn_router_span(l, s, sn)) { ferr = dense.last_error(); return false; }
                    }
                    if (B > d0) g_routes.add(l, dense.batch_ids() + (size_t) d0 * top_k, B - d0);
                    const double t1 = now_ms();
                    fd_dense += t1 - t0;
                    // GPU: the tier writes the routed sums straight into the dense half's routed tensor (no 2 x n*D*4
                    // host round trip per layer); CPU / STRATA_ROUTED_HOST=1: through the host as before
                    float * rdev = routed_host ? nullptr : dense.routed_dev();
                    if (l < 2) vram_log("pre-tier", l);
                    for (int s = d0; s < B; s += piece) {   // the experts for the dense rows (in pieces if the buffers cap)
                        const int pn = std::min(piece, B - s);
                        const bool ok = rdev ? tier.run_chunk_dev(l, pn, dense.batch_ids() + (size_t) s * top_k,
                                                                  dense.batch_wts() + (size_t) s * top_k,
                                                                  dense.batch_fn() + (size_t) s * n_embd,
                                                                  rdev + (size_t) s * n_embd)
                                             : tier.run_chunk(l, pn, dense.batch_ids() + (size_t) s * top_k,
                                                              dense.batch_wts() + (size_t) s * top_k,
                                                              dense.batch_fn() + (size_t) s * n_embd,
                                                              brouted.data() + (size_t) s * n_embd);
                        if (!ok) { ferr = "tier run refused"; return false; }
                    }
                    t0 = now_ms();
                    fd_tier += t0 - t1;
                    if (!rdev && !dense.upload_routed(brouted.data(), B)) { ferr = dense.last_error(); return false; }
                    for (int s = d0; s < B; s += SB) {
                        const int sn = std::min(SB, B - s);
                        if (!dense.finish_span(l, s, sn)) { ferr = dense.last_error(); return false; }
                    }
                    // serve DSpark seed (bcap_n > 0): the main_hidden rows of the batch's last bcap_n tokens - pass_v's
                    // HC-mean of the state leaving dm_layers[j] - read here, so the tail need not re-run as passes
                    if (bcap_n > 0 && c0 + B == n) {
                        for (int j = 0; j < 3; ++j) {
                            if (l != dm_layers[j]) continue;
                            const int cn = std::min(bcap_n, B);
                            bcap_got = cn;
                            if (!dense.read_state_cols(B - cn, cn, vx)) { ferr = dense.last_error(); return false; }
                            const size_t HC = vx.size() / ((size_t) cn * n_embd);
                            for (int t = 0; t < cn; ++t) {
                                // rows [0, cn): the consumer reads them from the start (they used to sit at the END of the
                                // bcap_n rows, so a final batch narrower than bcap_n seeded the drafter from zeros)
                                float * dst = bcap_vh.data() + (size_t) t * 3 * n_embd + (size_t) j * n_embd;
                                for (size_t h = 0; h < HC; ++h) {
                                    const float * xs = vx.data() + ((size_t) t * HC + h) * n_embd;
                                    for (int64_t d = 0; d < n_embd; ++d) dst[d] += xs[d] / (float) HC;
                                }
                            }
                        }
                    }
                }
                fd_dense += now_ms() - t0;
                const double tl = now_ms();
                if (all_logits) {   // every row, one sub-batch at a time (a B-wide head would need a ~2 GiB buffer)
                    all_logits->resize((size_t) (c0 + B) * (size_t) nv);
                    for (int s = 0; s < B; s += SB) {
                        const int sn = std::min(SB, B - s);
                        if (!dense.logits(s, sn, all_logits->data() + (size_t) (c0 + s) * nv)) { ferr = dense.last_error(); return false; }
                    }
                    if (c0 + B == n) std::copy(all_logits->end() - nv, all_logits->end(), last_logits.begin());
                } else if (c0 + B == n && !dense.logits(B - 1, 1, last_logits.data())) { ferr = dense.last_error(); return false; }
                for (int s = 0; tail_keep > 0 && s < B; s += SB)   // one sub-batch at a time, as above
                    if (!capture_tail(n, c0 + s, std::min(SB, B - s), s)) return false;
                fd_logits += now_ms() - tl;
                c0 += B;
                if (feed_cancel && c0 < n && feed_cancel()) return give_up(c0, false);
            } else {
                // ---- the sequential chunk path (unchanged): one pass of up to NP tokens through every layer
                const int m = std::min(NP, n - c0);
                double t0 = now_ms();
                if (!dense.begin(ids + c0, m, pos0 + c0)) { ferr = dense.last_error(); return false; }
                for (int l = 0; l < n_layer; ++l) {
                    const float * fn = nullptr;
                    if (!dense.attn_router(l, cids.data(), cw.data(), &fn)) { ferr = dense.last_error(); return false; }
                    g_routes.add(l, cids.data(), m);
                    if (dump_dir) {
                        char nb[64];
                        std::snprintf(nb, sizeof(nb), "L%02d_fn.f32", l);
                        dumpf(nb, fn, (size_t) m * (size_t) n_embd * 4);
                        std::vector<int> sel; std::vector<float> smask; int K = 0;
                        if (dense.read_sel(l, sel, smask, &K) && K > 0 && !sel.empty()) {
                            std::snprintf(nb, sizeof(nb), "L%02d_sel.i32", l);
                            dumpf(nb, sel.data(), sel.size() * 4);
                        }
                    }
                    const double t1 = now_ms();
                    fd_dense += t1 - t0;
                    const bool ok = m == 1 ? tier.run(l, cids.data(), cw.data(), fn, crouted.data())
                                           : tier.run_chunk(l, m, cids.data(), cw.data(), fn, crouted.data());
                    if (!ok) { ferr = "tier run refused"; return false; }
                    if (dump_dir) {
                        char nb[64];
                        std::snprintf(nb, sizeof(nb), "L%02d_routed.f32", l);
                        dumpf(nb, crouted.data(), (size_t) m * (size_t) n_embd * 4);
                    }
                    t0 = now_ms();
                    fd_tier += t0 - t1;
                    if (!dense.finish(l, crouted.data())) { ferr = dense.last_error(); return false; }
                }
                fd_dense += now_ms() - t0;
                const double tl = now_ms();
                if (all_logits) {
                    all_logits->resize((size_t) (c0 + m) * (size_t) nv);
                    if (!dense.logits(0, m, all_logits->data() + (size_t) c0 * nv)) { ferr = dense.last_error(); return false; }
                    if (c0 + m == n) std::copy(all_logits->end() - nv, all_logits->end(), last_logits.begin());
                } else if (c0 + m == n && !dense.logits(m - 1, 1, last_logits.data())) { ferr = dense.last_error(); return false; }
                if (!capture_tail(n, c0, m, 0)) return false;
                fd_logits += now_ms() - tl;
                c0 += m;
                if (feed_cancel && c0 < n && feed_cancel()) return give_up(c0, false);
            }
            progress(c0);
        }
        feed_done = n;
        if (dc.elastic_cols) {   // back to S columns
            if (!dense.set_columns(0)) { ferr = dense.last_error(); return false; }
            if (rdev1) rdev1 = dense.routed_dev();
        }
        tier.release_chunk();
        return true;
    };

    // ---- serve
    if (a.serve) {
        std::mutex qm;
        std::condition_variable qc;
        std::deque<std::string> q;
        bool eof = false;
        std::thread([&]() {
            std::string ln;
            while (std::getline(std::cin, ln)) {
                std::lock_guard<std::mutex> lk(qm);
                q.push_back(ln);
                qc.notify_all();
            }
            std::lock_guard<std::mutex> lk(qm);
            eof = true;
            qc.notify_all();
        }).detach();
        auto next_line = [&](std::string & out) -> bool {   // idle: wait for a line
            std::unique_lock<std::mutex> lk(qm);
            qc.wait(lk, [&] { return !q.empty() || eof; });
            if (q.empty()) return false;
            out = q.front();
            q.pop_front();
            return true;
        };
        auto try_line = [&](std::string & out) -> bool {    // between two passes: whatever has arrived, if anything
            std::lock_guard<std::mutex> lk(qm);
            if (q.empty()) return false;
            out = q.front();
            q.pop_front();
            return true;
        };
        auto at_eof = [&]() -> bool { std::lock_guard<std::mutex> lk(qm); return eof && q.empty(); };
        auto stop_pending = [&]() -> bool {   // a STOP line anywhere in the queue (consumed): the request being read is cancelled
            std::lock_guard<std::mutex> lk(qm);
            for (auto it = q.begin(); it != q.end(); ++it)
                if (*it == "STOP") { q.erase(it); return true; }
            return false;
        };
        feed_cancel = stop_pending;
        auto is_stop = [&](int t) { return std::find(a.stop.begin(), a.stop.end(), t) != a.stop.end() || t == (int) G.eos; };
        const int ctxv = (int) ctx;
        // ---- the LIVE conversation: the one whose rows and rings are on the device.  (A request parked in the middle of
        // its answer - several run at once, see the slots below - carries these in its Req meanwhile.)
        std::vector<int> hist;
        int snap_len = -1;                 // its prompt's end (Ds41Dense::snapshot there): a regen / tail resume restarts there
        int live_end = 0;                  // where a continuation restarts: hist.size(), or one less when the last verify pass
                                           // left an open compressor group at an odd end (see end_req)
        std::vector<float> snap_logits;    // the last prompt row's logits at the snapshot (a repeated prompt samples from them)
        // s6: DSpark's seed rows (main_hidden) of the snapshot's prompt tail - a repeated prompt (regenerate / a client
        // retry) feeds no new tokens, so without them DSpark had nothing to seed from and the whole answer decoded
        // without drafts (measured: 21.9 vs 31.8 tok/s on the same 1200-token edit).  cur_seed = this prompt's tail.
        std::vector<float> snap_seed, cur_seed;
        int snap_seed_n = 0, cur_seed_n = 0;
        // the prompt-end snapshot rings as host bytes (Ds41Dense::rings_out at the snapshot), and whether the device's
        // snapshot rings still hold them: a park of the conversation's END or of a request's decode position reuses
        // Ds41Dense::snapshot, so before a regen / tail resume they are put back (ensure_snap_rings)
        std::vector<uint8_t> snap_rings;
        bool snap_rings_dev = false;
        int n_req = 0;
        // STRATA_SERVE_PARK (default on): an agent harness sends short UNRELATED requests between the turns of its main
        // conversation (titles, summaries).  Each wrote its own positions from 0 over the main conversation's caches, so
        // the next main turn re-read the whole prompt (2026-10-15 harness: a 454-token side call -> the 20,736-token
        // system prompt again, 69.5 s).  Now the main conversation is parked (a host copy) and put back when a request
        // continues it.  STRATA_SERVE_PARK_MIN: only conversations of >= this many tokens (default 2048).
        const bool park_on = !(std::getenv("STRATA_SERVE_PARK") && std::getenv("STRATA_SERVE_PARK")[0] == '0');
        // STRATA_SERVE_TAIL_RESUME (default on, =0 off): a request whose prompt leaves the snapshot's within its last
        // dense.rollback_max() tokens restarts there instead of re-reading everything (see the reuse rules below)
        const bool tail_resume = !(std::getenv("STRATA_SERVE_TAIL_RESUME") && std::getenv("STRATA_SERVE_TAIL_RESUME")[0] == '0');
        const int park_min = std::getenv("STRATA_SERVE_PARK_MIN") ? std::atoi(std::getenv("STRATA_SERVE_PARK_MIN")) : 2048;
        // 2026-10-22, several agents at once (a parent and its subagents on this one engine): the old park kept ONE
        // conversation and only the rows one unrelated request could write (prompt + max tokens), so it was dropped as soon as
        // the next request of the other conversation reached a few hundred positions further - dsh asks for 256K output tokens
        // - and every switch re-read a whole prompt (10K tokens 45 s, a 190K conversation 8 minutes).  Now every parked
        // conversation is a COMPLETE host copy (Ds41Dense::park_out: every row up to its END and the rings there, ~0.9 KB a
        // token, 170 MB at 190K) plus its prompt-end rings and logits, up to STRATA_SERVE_PARK_SLOTS (default 12) of them
        // within STRATA_SERVE_PARK_GIB (default 8); the least recently used goes first.  A request that continues one swaps it
        // in (the live conversation is parked first) - at its END (the next turn of a conversation reads only the new text,
        // not the answer again), at its prompt end (the same prompt again, or a cancel-and-resend that leaves it within
        // rollback_max() tokens), or within rollback_max() of its end.
        // 2026-10-22 evening, SHARED PREFIXES: every subagent of an agent harness starts with the same system prompt + tools
        // (dsh: ~9.3K tokens, 44 s to read), and every one read it in full - the parked conversations hold it, but a resume
        // needs the window rings AT the point of divergence, which no park had.  Now, when a prompt shares at least
        // STRATA_SERVE_PREFIX_MIN tokens (default 1024, 0 = off) beyond what it resumes from with any conversation the engine
        // knows, its read is split there and the state at that point is parked as a SHARED PREFIX: served by copy (never
        // consumed), kept while it is used (up to STRATA_SERVE_PREFIX_SLOTS, default 4, least recently used first), evicted
        // after every ordinary park.  The next request that starts with it restarts right there.
        struct ParkEntry {
            Ds41Dense::ParkedPtr p;              // every row up to hist.size() + the rings at that END (snapshot pos = hist.size())
            std::vector<int> hist;               // the prompt, and what was decoded after it (all but a token that would start an open group)
            int snap_len = 0;                    // the prompt's end (== hist.size() when nothing was decoded, and for a prefix)
            std::vector<uint8_t> prompt_rings;   // the rings at snap_len (empty when snap_len == hist.size(): the END rings are those)
            std::vector<float> logits;           // the last prompt row's logits at snap_len
            std::vector<float> seed;             // DSpark seed rows of the prompt's tail (a regen re-seeds the drafter from them)
            int seed_n = 0;
            std::vector<uint8_t> dsp_win;        // the drafter window at the END: a continuation keeps its drafts (empty: none saved)
            std::vector<int> dsp_filled;
            bool prefix = false;                 // a shared prefix (see above)
            uint64_t stamp = 0;                  // when parked / last used (LRU)
            int hits = 0;
            std::string file;                    // the shared-prefix file it came from / went to (STRATA_SERVE_PREFIX_DIR; "" = none)
        };
        std::vector<ParkEntry> parks;
        uint64_t park_clock = 0;
        const size_t park_slots = std::getenv("STRATA_SERVE_PARK_SLOTS") ? (size_t) std::max(0, std::atoi(std::getenv("STRATA_SERVE_PARK_SLOTS"))) : 12;
        const double park_cap = (std::getenv("STRATA_SERVE_PARK_GIB") ? std::atof(std::getenv("STRATA_SERVE_PARK_GIB")) : 8.0) * 1073741824.0;
        const int prefix_min = std::getenv("STRATA_SERVE_PREFIX_MIN") ? std::atoi(std::getenv("STRATA_SERVE_PREFIX_MIN")) : 1024;
        const size_t prefix_slots = std::getenv("STRATA_SERVE_PREFIX_SLOTS") ? (size_t) std::max(0, std::atoi(std::getenv("STRATA_SERVE_PREFIX_SLOTS"))) : 4;
        auto parks_bytes = [&]() -> double {
            double b = 0;
            for (const auto & e : parks) b += (double) Ds41Dense::parked_bytes(e.p) + (double) e.prompt_rings.size() + (double) e.seed.size() * 4;
            return b;
        };
        // STRATA_SERVE_PREFIX_DIR=<dir> (OPT-IN, unset = off): shared-prefix parks are also kept as files there (one per
        // prefix: its token ids, the last-row logits, the DSpark seed rows and the park's rows, ~20 MiB at 9K tokens;
        // 0600 in a 0700 directory) and loaded at the next start, so a restart does not cost each prefix's read again.
        // A file holds the prefix's TOKENS - the harness's system prompt - so the user decides whether to turn it on.
        // The files are keyed by the model file (size + mtime), the context size and the cache layout; one that does
        // not match this engine is skipped.  Dropped prefixes take their file along.
        const char * prefix_dir = std::getenv("STRATA_SERVE_PREFIX_DIR");
        if (prefix_dir && !*prefix_dir) prefix_dir = nullptr;
        uint64_t model_size = 0, model_mtime = 0;
        if (prefix_dir) {
            std::error_code ec;
            model_size = (uint64_t) std::filesystem::file_size(a.model, ec);
            model_mtime = (uint64_t) std::filesystem::last_write_time(a.model, ec).time_since_epoch().count();
            std::filesystem::create_directories(prefix_dir, ec);
            std::filesystem::permissions(prefix_dir, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace, ec);
        }
        auto prefix_file_name = [&](const std::vector<int> & ids) -> std::string {   // FNV-1a 64 of the ids
            uint64_t h = 1469598103934665603ull;
            for (int t : ids) for (int k = 0; k < 4; ++k) { h ^= (uint64_t) ((((uint32_t) t) >> (8 * k)) & 255u); h *= 1099511628211ull; }
            char nb[48];
            std::snprintf(nb, sizeof(nb), "/pfx-%016llx.bin", (unsigned long long) h);
            return std::string(prefix_dir) + nb;
        };
        auto prefix_file_save = [&](ParkEntry & e) {
            if (!prefix_dir || !e.prefix) return;
            std::vector<uint8_t> blob;
            if (!dense.park_to_bytes(e.p, blob)) return;
            const std::string path = prefix_file_name(e.hist), tmp = path + ".tmp";
            {
                std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
                if (!f) { std::fprintf(stderr, "serve: prefix file: cannot write %s\n", tmp.c_str()); return; }
                auto w64 = [&](uint64_t v) { f.write((const char *) &v, 8); };
                f.write("DS41PFX1", 8);
                w64(model_size); w64(model_mtime); w64((uint64_t) ctxv);
                w64((uint64_t) e.hist.size()); w64((uint64_t) e.logits.size()); w64((uint64_t) n_embd); w64((uint64_t) e.seed_n);
                w64((uint64_t) e.hist.size() + 64); w64((uint64_t) blob.size());
                f.write((const char *) e.hist.data(), (std::streamsize) (e.hist.size() * 4));
                f.write((const char *) e.logits.data(), (std::streamsize) (e.logits.size() * 4));
                f.write((const char *) e.seed.data(), (std::streamsize) (e.seed.size() * 4));
                f.write((const char *) blob.data(), (std::streamsize) blob.size());
                if (!f) { std::fprintf(stderr, "serve: prefix file: write failed: %s\n", tmp.c_str()); return; }
            }
            std::error_code ec;
            std::filesystem::permissions(tmp, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write, std::filesystem::perm_options::replace, ec);
            std::filesystem::rename(tmp, path, ec);
            if (ec) { std::fprintf(stderr, "serve: prefix file: rename failed: %s\n", ec.message().c_str()); return; }
            e.file = path;
            std::fprintf(stderr, "serve: shared prefix of %zu tokens saved to %s (%.0f MiB)\n", e.hist.size(), path.c_str(),
                         (double) std::filesystem::file_size(path, ec) / 1048576.0);
        };
        auto prefix_file_drop = [&](const ParkEntry & e) {
            if (e.file.empty()) return;
            std::error_code ec;
            std::filesystem::remove(e.file, ec);
        };
        auto prefix_files_load = [&]() {
            if (!prefix_dir) return;
            std::error_code ec;
            std::vector<std::pair<std::filesystem::file_time_type, std::string>> files;
            for (const auto & de : std::filesystem::directory_iterator(prefix_dir, ec)) {
                const std::string nm = de.path().filename().string();
                if (nm.rfind("pfx-", 0) == 0 && nm.size() > 8 && nm.compare(nm.size() - 4, 4, ".bin") == 0)
                    files.emplace_back(de.last_write_time(ec), de.path().string());
            }
            std::sort(files.begin(), files.end(), [](const auto & x, const auto & y) { return x.first > y.first; });
            size_t loaded = 0;
            for (const auto & fe : files) {
                if (loaded >= prefix_slots) break;
                std::ifstream f(fe.second, std::ios::binary);
                char magic[8] = {0};
                uint64_t h[9] = {0};
                f.read(magic, 8);
                f.read((char *) h, 72);
                if (!f || std::memcmp(magic, "DS41PFX1", 8) != 0 || h[0] != model_size || h[1] != model_mtime || h[2] != (uint64_t) ctxv ||
                    h[4] != (uint64_t) nv || h[5] != (uint64_t) n_embd || h[3] < 16 || h[3] > (uint64_t) ctxv || h[7] != h[3] + 64) {
                    std::fprintf(stderr, "serve: prefix file skipped (another model / context / engine): %s\n", fe.second.c_str());
                    continue;
                }
                ParkEntry e;
                e.hist.resize((size_t) h[3]);
                e.logits.resize((size_t) h[4]);
                e.seed.resize((size_t) h[6] * 3 * (size_t) n_embd);
                e.seed_n = (int) h[6];
                std::vector<uint8_t> blob((size_t) h[8]);
                f.read((char *) e.hist.data(), (std::streamsize) (e.hist.size() * 4));
                f.read((char *) e.logits.data(), (std::streamsize) (e.logits.size() * 4));
                f.read((char *) e.seed.data(), (std::streamsize) (e.seed.size() * 4));
                f.read((char *) blob.data(), (std::streamsize) blob.size());
                std::string perr;
                if (!f || !(e.p = dense.park_from_bytes(blob.data(), blob.size(), (int64_t) h[7], perr))) {
                    std::fprintf(stderr, "serve: prefix file skipped (%s): %s\n", perr.empty() ? "short" : perr.c_str(), fe.second.c_str());
                    continue;
                }
                e.snap_len = (int) e.hist.size();
                e.prefix = true;
                e.stamp = 0;   // older than anything parked in this run: used ones move up, unused ones go first
                e.file = fe.second;
                parks.push_back(std::move(e));
                ++loaded;
            }
            if (loaded) std::fprintf(stderr, "serve: %zu shared prefix(es) loaded from %s\n", loaded, prefix_dir);
        };
        auto evict_parks = [&](double need) {   // room for one more entry of `need` bytes: the least recently used first, prefixes last
            while (!parks.empty() && (parks.size() >= park_slots || parks_bytes() + need > park_cap)) {
                auto pick = parks.end();
                for (auto it = parks.begin(); it != parks.end(); ++it)
                    if (!it->prefix && (pick == parks.end() || it->stamp < pick->stamp)) pick = it;
                if (pick == parks.end())
                    for (auto it = parks.begin(); it != parks.end(); ++it)
                        if (pick == parks.end() || it->stamp < pick->stamp) pick = it;
                std::fprintf(stderr, "serve: park full - dropped the %zu-token %s\n", pick->hist.size(), pick->prefix ? "shared prefix" : "conversation");
                prefix_file_drop(*pick);
                parks.erase(pick);
            }
        };
        auto ensure_snap_rings = [&]() {   // the device's snapshot rings = the live conversation's prompt-end rings (regen / tail resume)
            if (!snap_rings_dev && snap_len > 0 && !snap_rings.empty() && dense.rings_in(snap_rings, snap_len)) snap_rings_dev = true;
        };
        bool cache_open = !defer_vram;   // deferred: the arena only until the first decode
#if defined(DS41_HAVE_DSPARK)
        // STRATA_SERVE_DSPARK: DSpark's window (main_hidden of the last <= 128 positions) is seeded by running a
        // request's last min(128, new) prompt tokens through pass_v instead of the passes / batch (no extra work), and
        // it carries over when the next request continues this one (dsp_next = the position its ring expects next);
        // any other reuse resets it (lower acceptance for a while, never a different output) - unless the park kept the
        // window (ParkEntry::dsp_win), which a continuation at the END puts back.
        int dsp_next = -1;
        const int sp_depth_serve = std::getenv("STRATA_DSPARK_DEPTH") ? std::max(1, std::min(5, std::atoi(std::getenv("STRATA_DSPARK_DEPTH")))) : 4;
        long sv_pass = 0, sv_acc = 0;
#endif
        // the conversation on the device into a park slot: its rows up to its END and the rings there, its prompt-end rings
        // (a regen / tail resume at the prompt), the drafter window.  `end_len` = live_end: where a continuation restarts.
        auto park_conv = [&](int end_len) -> bool {
            if (!park_on || park_slots == 0 || snap_len < park_min || hist.size() < (size_t) snap_len || end_len < snap_len ||
                end_len > (int) hist.size()) return false;
            if (!dense.snapshot(end_len)) return false;   // the live rings are the END's
            snap_rings_dev = false;
            Ds41Dense::ParkedPtr p = dense.park_out((int64_t) end_len + 64);
            if (!p) return false;
            const double need = (double) Ds41Dense::parked_bytes(p) + (double) snap_rings.size() + (double) snap_seed.size() * 4;
            if (need > park_cap) { std::fprintf(stderr, "serve: the %d-token conversation (%.0f MiB) does not fit the park\n", end_len, need / 1048576.0); return false; }
            evict_parks(need);
            ParkEntry e;
            e.p = std::move(p);
            e.hist.assign(hist.begin(), hist.begin() + end_len);
            e.snap_len = snap_len;
            if (end_len != snap_len) e.prompt_rings = snap_rings;
            e.logits = snap_logits;
            e.seed = snap_seed;
            e.seed_n = snap_seed_n;
#if defined(DS41_HAVE_DSPARK)
            if (dsp && dsp_next == end_len) dsp->window_out(e.dsp_win, e.dsp_filled);
#endif
            e.stamp = ++park_clock;
            parks.push_back(std::move(e));
            std::fprintf(stderr, "serve: parked the %d-token conversation (prompt %d; %.0f MiB; %zu parked, %.0f MiB)\n", end_len, snap_len,
                         need / 1048576.0, parks.size(), parks_bytes() / 1048576.0);
            return true;
        };
        // STRATA_SERVE_STEP_MAX: new prompt tokens up to which a request runs on the decode side - passes of
        // STRATA_SERVE_PASS_M tokens with the VRAM cache and the decode biases (~40 tok/s) - instead of the exact
        // layer-major batch (~16.8 s + n / 250 tok/s; measured crossover ~765 tokens); 0 = always the batch (every prompt
        // token exact)
        const int step_max = std::getenv("STRATA_SERVE_STEP_MAX") ? std::atoi(std::getenv("STRATA_SERVE_STEP_MAX")) : 768;
        // STRATA_SERVE_PASS_BIAS=0: those passes route exactly (the decode biases only while decoding)
        const bool pass_bias = !(std::getenv("STRATA_SERVE_PASS_BIAS") && std::getenv("STRATA_SERVE_PASS_BIAS")[0] == '0');
        // ---- several requests at once: the server's batch slots (INFO batch_slots=N; BGEN <slot> ... / BT <slot> <id> /
        // BDONE <slot> ... / BSTOP <slot>, see serve/server.py generate_batched).  This engine holds ONE conversation on the
        // device, so the slots take turns: each runs STRATA_SERVE_SLICE verify passes (default 6, ~1.3 s), then its rows,
        // rings, drafter window and sampler state go to host memory and the next one's come back (~10-20 ms a switch).  Every
        // request streams from its start and none waits behind another's answer; the engine's total speed does not change
        // (a pass is bound by the expert bytes it streams, which a second sequence only adds to), and a prompt read still
        // runs alone (one layer-major batch).  STRATA_SERVE_SLOTS (default 4; 0 = one request at a time, GEN only).
        const int n_slots = std::getenv("STRATA_SERVE_SLOTS") ? std::max(0, std::min(16, std::atoi(std::getenv("STRATA_SERVE_SLOTS")))) : 4;
        const int slice_passes = std::max(1, std::getenv("STRATA_SERVE_SLICE") ? std::atoi(std::getenv("STRATA_SERVE_SLICE")) : 6);
        struct Req {
            int slot = -1;                     // -1: a solo request (GEN: T / DONE lines); >= 0: a batch slot (BT / BDONE)
            bool active = false, live = false, stop_req = false;
            // the conversation while it is not on the device (the live state above, moved here by park_req)
            std::vector<int> hist;
            int snap_len = -1, live_end = 0;
            std::vector<float> snap_logits, snap_seed;
            int snap_seed_n = 0;
            std::vector<uint8_t> snap_rings;
            int dsp_next = -1;
            Ds41Dense::ParkedPtr park;         // rows + the rings at the decode position
            std::vector<uint8_t> dsp_win;
            std::vector<int> dsp_filled;
            bool dsp_saved = false;
            // sampling and routing
            float temp = 0, top_p = 1, min_p = 0;
            int tk = 0;
            std::mt19937_64 rng;
            PenaltyState pen;
            float route_bias = 0, route_bias_file = 0;
            int rprot = 0, rpool = 0;
            // the request
            int max_new = 0, total = 0, reused = 0, gen = 0;
            double prompt_ms = 0, td0 = 0;
            const char * finish = "length";
            // the DSpark loop (as the one-shot loop: P = the last fed position, t = the token for P+1, sp_ids = drafts for P+2..)
            bool spec = false;
            int P = 0, t = -1, prev = 0;
            bool dirty = false;
            int sp_ids[5] = {0};
            float sp_conf[5] = {0};
            long rq_pass = 0, rq_acc = 0, lk_pass = 0, lk_dr = 0, lk_acc = 0;
            int lk_cool = 0, lk_back = 1;
            // the plain loop (no drafts): the next token to feed, its position
            std::vector<float> lg;
            int pos = 0, ptok = -1;
            int slice = 0;                     // passes run in the current turn
            // conversation slots (dslots > 1): the device slot holding this request's conversation while it is not live (-1: a
            // host park, r.park), the prompt-end rings flag that went with it, whether its drafter window fell behind, the
            // batch windows it decoded in, when it last ran (the eviction order), its admission number (dumps)
            int dslot = -1;
            bool snap_dev = false, dsp_stale = false;
            long w_pass = 0, stamp = 0;
            int id = 0;
        };
        std::vector<Req> reqs((size_t) n_slots + 1);   // the last one is the solo request
        for (size_t i = 0; i < reqs.size(); ++i) reqs[i].slot = i + 1 < reqs.size() ? (int) i : -1;
        Req * live_req = nullptr;              // the ACTIVE request whose conversation is on the device (null: a finished one, or none)
        auto any_active = [&]() -> bool { for (const auto & r : reqs) if (r.active) return true; return false; };
        // ---- conversation slots on the device (Ds41Dense::n_slots; 1 = the engine as before).  The globals above (hist,
        // snap_*, live_end) always describe the BOUND slot's conversation; a running request that is not live keeps its own in
        // Req (host side) + its slot (device side, r.dslot >= 0) or a host park (r.dslot < 0).  A finished conversation is
        // only known on the slot it ended on: leaving that slot parks it in host memory (park_conv), as an unrelated
        // request would.
        const int dslots = dense.n_slots();
        long run_clock = 0;                                  // Req::stamp: the order requests last ran (eviction)
        long win_n = 0, win_rows = 0; double win_ms = 0;     // batch windows run, their rows, their pass wall time
        double win_post_ms = 0, win_dsp_ms = 0, win_slow_max = 0; long win_slow = 0;   // after the pass (sampling, the drafter), slow windows
        auto slot_owner = [&](int d) -> Req * {   // the running request whose conversation slot d holds (live or resident)
            for (auto & q : reqs) if (q.active && (q.live ? dense.slot() == d : q.dslot == d)) return &q;
            return nullptr;
        };
        auto free_slot = [&]() -> int {           // a slot no running request holds: the bound one first; -1 = all taken
            if (!slot_owner(dense.slot())) return dense.slot();
            for (int d = 0; d < dslots; ++d) if (!slot_owner(d)) return d;
            return -1;
        };
        // bind slot d for the single-sequence path (no live request): the finished conversation on the bound slot goes to a
        // host park first, the device conversation is then unknown (d's last one was parked the same way when it was left)
        auto switch_slot = [&](int d) -> bool {
            if (d == dense.slot()) return true;
            if (!live_req && snap_len > 0 && !hist.empty()) park_conv(live_end);
            hist.clear();
            snap_len = -1;
            live_end = 0;
            snap_rings_dev = false;
#if defined(DS41_HAVE_DSPARK)
            dsp_next = -1;
#endif
            if (!dense.bind_slot(d)) { ferr = dense.last_error(); return false; }
            return true;
        };
        // a resident request's conversation to a host park (its slot is needed): bind its slot, copy it out as park_req does
        auto evict_req = [&](Req & r) -> bool {
            if (r.live || r.dslot < 0 || !switch_slot(r.dslot)) return false;
            const int top = (int) r.hist.size() + 32;
            dense.snapshot(std::min(top, ctxv - 1));
            r.park = dense.park_out((int64_t) top + 64);
            r.dslot = -1;
            r.snap_dev = false;
            return r.park != nullptr;
        };
        // STRATA_SERVE_ROWDUMP=<dir> (exactness checks): every decode row's logits as {id, pos, fed token, n_vocab} + floats
        static FILE * rowdump_f = std::getenv("STRATA_SERVE_ROWDUMP")
            ? std::fopen((std::string(std::getenv("STRATA_SERVE_ROWDUMP")) + "/rows.bin").c_str(), "wb") : nullptr;
        auto rowdump = [&](int id, int pos, int tok, const float * lg) {
            if (!rowdump_f) return;
            const int32_t h[4] = { id, pos, tok, nv };
            std::fwrite(h, 4, 4, rowdump_f);
            std::fwrite(lg, 4, (size_t) nv, rowdump_f);
        };
        prefix_files_load();
        std::printf("INFO ctx=%d experts_vram=%lld arena_experts=%lld\n", ctxv, (long long) tier.resident(),
                    (long long) tier.arena_experts());
        if (n_slots > 0) std::printf("INFO batch_slots=%d\n", n_slots);
        std::printf("READY %d stop\n", ctxv);
        std::fflush(stdout);
        std::fprintf(stderr, "serve: ready (ctx %d, prefill chunk %d, %d request slots, slice %d passes, prefix parks >= %d tokens)\n",
                     ctxv, NP, n_slots, slice_passes, prefix_min);
        if (dslots > 1)
            std::fprintf(stderr, "serve: %d conversation slots on the device (%.0f MiB each): the requests running at once stay resident "
                                 "and decode in batch windows\n", dslots, (double) dense.slot_bytes() / 1048576.0);
        // per-request decode biases (GEN keys route_bias= / route_bias_file=, from the server's "strata_tune"): a request
        // that names neither gets the --route-bias / --route-bias-file the engine started with, so one server can answer
        // exact-routing requests (both 0) beside the fast default without a restart.  apply_bias() writes zeros over a
        // bias it set earlier, so an exact request never inherits the previous one's.
        const float cfg_route_bias = a.route_bias, cfg_route_bias_file = a.route_bias_file;
        // the smart router's ranks (route_protect= / route_pool=, engines built with STRATA_RB_SMART / _PROTECT / _POOL):
        // a request that names neither gets the env's
        auto env_rank = [](const char * k) { const char * e = std::getenv(k); return e ? std::max(0, std::atoi(e)) : 0; };
        const int cfg_rprot = env_rank("STRATA_RB_PROTECT"), cfg_rpool = env_rank("STRATA_RB_POOL");
        int cur_rprot = cfg_rprot, cur_rpool = cfg_rpool;
        bool previous_vision = false;
        // ---- the live request off the device (into r), and back
        auto park_req = [&](Req & r) {
            if (dslots > 1) {   // conversation slots: the state stays on the device, in this slot - only the host side moves
                r.dslot = dense.slot();
                r.snap_dev = snap_rings_dev;
                snap_rings_dev = false;
            } else {
                const int top = (int) hist.size() + 32;   // a verify pass may have written rows past the accepted ones
                dense.snapshot(std::min(top, ctxv - 1));
                snap_rings_dev = false;
                r.park = dense.park_out((int64_t) top + 64);
            }
#if defined(DS41_HAVE_DSPARK)
            r.dsp_saved = dsp && dsp->window_out(r.dsp_win, r.dsp_filled);
            r.dsp_next = dsp_next;
            dsp_next = -1;
#endif
            r.hist = std::move(hist);
            hist.clear();
            r.snap_len = snap_len;
            r.live_end = live_end;
            snap_len = -1;
            live_end = 0;
            r.snap_logits = std::move(snap_logits);
            r.snap_seed = std::move(snap_seed);
            r.snap_seed_n = snap_seed_n;
            r.snap_rings = std::move(snap_rings);
            snap_rings.clear();
            r.rng = rng;
            r.pen = g_pen;
            r.live = false;
            live_req = nullptr;
        };
        auto make_live = [&](Req & r) -> bool {
            if (r.live) return true;
            if (live_req) park_req(*live_req);
            const bool resident = dslots > 1 && r.dslot >= 0;
            if (resident) {   // its conversation is in its slot: bind it, nothing to copy
                if (!switch_slot(r.dslot)) return false;
                if (!hist.empty()) { if (snap_len > 0) park_conv(live_end); hist.clear(); snap_len = -1; live_end = 0; }
            } else {
                if (dslots > 1) {   // from a host park into a free slot (the least recently run resident request's when all are taken)
                    int d = free_slot();
                    if (d < 0) {
                        Req * victim = nullptr;
                        for (auto & q : reqs)
                            if (q.active && !q.live && q.dslot >= 0 && &q != &r && (!victim || q.stamp < victim->stamp)) victim = &q;
                        if (!victim || !evict_req(*victim)) { ferr = "no conversation slot for the parked request"; return false; }
                        d = dense.slot();
                    }
                    if (!switch_slot(d)) return false;
                }
                if (snap_len > 0 && !hist.empty()) park_conv(live_end);   // a finished conversation: kept for its next turn
                hist.clear();
                snap_len = -1;
                if (!r.park || !dense.park_in(r.park) || !dense.restore()) { ferr = "the parked request could not be put back"; return false; }
                r.park.reset();
            }
            r.stamp = ++run_clock;
#if defined(DS41_HAVE_DSPARK)
            if (dsp && r.dsp_saved) dsp->window_in(r.dsp_win, r.dsp_filled);
            dsp_next = r.dsp_next;
#endif
            hist = std::move(r.hist);
            snap_len = r.snap_len;
            live_end = r.live_end;
            snap_logits = std::move(r.snap_logits);
            snap_seed = std::move(r.snap_seed);
            snap_seed_n = r.snap_seed_n;
            snap_rings = std::move(r.snap_rings);
            snap_rings_dev = resident && r.snap_dev;   // a resident slot's device snapshot rings are as it left them
            rng = r.rng;
            g_pen = r.pen;
#if defined(DS41_HAVE_DSPARK)
            if (dsp && r.dsp_stale) {   // its drafter window fell behind in the batch windows: start the window over, no drafts
                dsp->reset();           // until the next pass refills it (the first drafts would be guesses from an old window)
                for (float & c : r.sp_conf) c = -1.0e9f;
                r.dsp_stale = false;
            }
#endif
            if (!hist.empty()) dense.prime_history(hist.data(), std::max<int64_t>(0, (int64_t) hist.size() - 16), (int64_t) hist.size());
            if (r.route_bias != a.route_bias || r.route_bias_file != a.route_bias_file) {
                a.route_bias = r.route_bias;
                a.route_bias_file = r.route_bias_file;
                bias_now = true;
                apply_bias();
            }
            if ((r.rprot != cur_rprot || r.rpool != cur_rpool) && dense.set_route_select(r.rprot, r.rpool)) {
                cur_rprot = r.rprot;
                cur_rpool = r.rpool;
            }
            r.live = true;
            live_req = &r;
            return true;
        };
        auto emit_tok = [&](Req & r, int tok) -> bool {   // false = the request ends with this token
            if (r.slot >= 0) std::printf("BT %d %d\n", r.slot, tok); else std::printf("T %d\n", tok);
            std::fflush(stdout);
            ++r.gen;
            if (is_stop(tok)) { r.finish = "stop"; return false; }
            if (r.max_new > 0 && r.gen >= r.max_new) return false;
            return true;
        };
        // the request is over (it is live): its figures, the conversation's END, the DONE / BDONE line
        auto end_req = [&](Req & r, bool at_admission) {
            const double dec_ms = now_ms() - r.td0;
            const Ds41MoeStats st = tier.stats();
            if (std::strcmp(r.finish, "error") == 0) {
                std::fprintf(stderr, "serve: decode failed: %s\n", ferr.c_str());
                hist.clear();
                snap_len = -1;
                live_end = 0;
                r.finish = "length";
#if defined(DS41_HAVE_DSPARK)
                dsp_next = -1;
#endif
            } else {
                // where a continuation restarts: after an open compressor group (a verify pass with rejected drafts fed rows past
                // the accepted ones, and the next position is odd) the token that starts the group is fed again - as the loop's
                // own refeed does
                live_end = (r.spec && r.dirty && (hist.size() & 1)) ? (int) hist.size() - 1 : (int) hist.size();
#if defined(DS41_HAVE_DSPARK)
                if (r.spec && !r.dsp_stale) dsp_next = live_end;   // the ring holds every position <= P (the last emitted token is not fed)
#endif
            }
            std::fprintf(stderr, "serve: %d tokens in %.2f s = %.2f tok/s (%s%s); experts: %lld hit, %lld cpu, %lld pcie, %lld file%s\n",
                         r.gen, dec_ms / 1000.0, 1000.0 * std::max(0, r.gen - 1) / std::max(dec_ms, 1e-9), r.finish,
                         r.slot >= 0 ? (", slot " + std::to_string(r.slot)).c_str() : "",
                         (long long) st.hits, (long long) st.cpu, (long long) st.pcie, (long long) st.file_tier,
                         st.skipped > 0 ? (", skipped " + std::to_string(st.skipped)).c_str() : "");
            if (const Ds41Dense::RouteMeter rm = dense.route_meter(false); rm.on && rm.slots > 0)
                std::fprintf(stderr, "serve: routing vs exact: %.2f%% of expert picks swapped, %.2f%% of router weight kept (%lld slots)\n",
                             100.0 * (double) rm.swaps / (double) rm.slots, 100.0 * rm.weight_kept, rm.slots);
            if (r.w_pass > 0)
                std::fprintf(stderr, "serve: batch windows: %ld of this request's passes; %ld windows so far, %.2f rows each, %.0f ms a window + %.0f ms after it "
                             "(drafter %.0f); %ld slow windows (max %.0f ms)\n", r.w_pass, win_n, win_n ? (double) win_rows / (double) win_n : 0.0,
                             win_n ? win_ms / (double) win_n : 0.0, win_n ? win_post_ms / (double) win_n : 0.0, win_n ? win_dsp_ms / (double) win_n : 0.0,
                             win_slow, win_slow_max);
            if (rowdump_f) std::fflush(rowdump_f);
            if (vram_trace) {   // who holds the card while decoding (the expert slots are the rest)
                size_t tf = 0, tt = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(be), &tf, &tt);
                double dsp_gib = 0.0;
#if defined(DS41_HAVE_DSPARK)
                if (dsp) dsp_gib = (double) dsp->weight_bytes() / 1073741824.0;
#endif
                std::fprintf(stderr, "vram ledger: %s; DSpark weights %.2f GiB; free %.2f of %.2f GiB\n",
                             dense.vram_ledger().c_str(), dsp_gib, tf / 1073741824.0, tt / 1073741824.0);
            }
#if defined(DS41_HAVE_DSPARK)
            if (r.lk_pass > 0)
                std::fprintf(stderr, "serve: lookup %ld/%ld drafts accepted in %ld passes\n", r.lk_acc, r.lk_dr, r.lk_pass);
            sv_pass += r.rq_pass; sv_acc += r.rq_acc;
            if (r.rq_pass > 0) {
                if (dsp) dsp->write_counts();
                std::fprintf(stderr, "serve: DSpark %ld passes, %.2f tokens/pass (all requests %.2f)\n", r.rq_pass,
                             1.0 + (double) r.rq_acc / r.rq_pass, 1.0 + (double) sv_acc / std::max(1L, sv_pass));
            }
#endif
            r.active = false;
            r.live = false;
            live_req = nullptr;
            tier.rebalance_tick(!any_active());   // the gap until the next request: retire a large batch for the reader
            if (r.slot >= 0) {
                if (!at_admission) std::printf("BDONE %d %d %s %.1f\n", r.slot, r.gen, r.finish, dec_ms);
            } else {
                std::printf("DONE %d %d %.1f %.1f %s 0 0 %d %lld %lld 0 %lld 0 %d\n", r.gen, r.total, r.prompt_ms, dec_ms, r.finish, r.reused,
                            (long long) st.hits, (long long) st.lookups(), (long long) st.file_tier, r.total - r.reused);
            }
            std::fflush(stdout);
            if (g_routes.on()) {   // after DONE: the client already has its answer
                std::string rerr;
                if (!g_routes.write(rerr)) std::fprintf(stderr, "routes: %s\n", rerr.c_str());
            }
        };
#if defined(DS41_HAVE_DSPARK)
        // ---- DSpark speculative sampling, one verify pass of the live request r (as the one-shot loop: P = last fed position,
        // t = the token for P+1, sp_ids = drafts for P+2..; each verify row's distribution is sample_p's, spec_accept keeps it
        // exact).  false = the request ended (r.finish says how).
        // STRATA_SERVE_LOOKUP=L (s6, opt-in until measured; mab776/Strata-DS4 M11's design): when the context's
        // last G+1 tokens (STRATA_LOOKUP_NGRAM=G, default 3) occurred earlier in this request (prompt + output),
        // the tokens that followed them there are the verify pass's drafts instead of DSpark's - an agent's output
        // copies its context (edits, rewrites, quoted tool results).  spec_accept treats every draft as a
        // deterministic token (accept with p, else resample without it), so the output distribution is unchanged.
        // A match of M tokens drafts M - G + 1 (<= L, <= the pass width).  STRATA_LOOKUP_AGREE (default 1): only
        // when the copy's first token is DSpark's own first draft.  A lookup pass with less than half accepted
        // sits the lookup out 1, 2, 4 .. 64 passes.  DSpark steps on the verified rows either way.
        // kVerify (s6): rows one verify pass may take - 16, as the prompt passes (pass_v splits the tier calls at
        // kMaxMulti - 8 then, 16 now - and the pass buffers hold 17).  A copy is accepted ~96% on edits, so the
        // draft length was the limit at the old kMaxMulti - 2 = 6 (s6: 932/972 accepted in 169 passes, only +3%
        // over DSpark's own).
        constexpr int kVerify = 16;
        static const int lk_max = std::getenv("STRATA_SERVE_LOOKUP") ? std::max(0, std::min(kVerify - 2, std::atoi(std::getenv("STRATA_SERVE_LOOKUP")))) : 0;
        static const int lk_g = std::getenv("STRATA_LOOKUP_NGRAM") ? std::max(1, std::atoi(std::getenv("STRATA_LOOKUP_NGRAM"))) : 3;
        static const bool lk_agree = !(std::getenv("STRATA_LOOKUP_AGREE") && std::getenv("STRATA_LOOKUP_AGREE")[0] == '0');
        // STRATA_DSPARK_CONF=<tau>, as the one-shot loop: verify only the leading drafts whose confidence-head
        // probability is >= tau (the emitted tokens are still the trunk's - only the verify width changes)
        static const float conf_tau_s = std::getenv("STRATA_DSPARK_CONF") ? (float) std::atof(std::getenv("STRATA_DSPARK_CONF")) : -1.0f;
        std::vector<int> vt;
        std::vector<std::pair<int, double>> dist;
        auto run_pass = [&](Req & r) -> bool {
            if (r.stop_req) { r.finish = "cancel"; return false; }
            int k = sp_depth_serve;
            if (r.max_new > 0) k = std::min(k, r.max_new - r.gen);
            k = std::min(k, ctxv - (r.P + 3));   // the pass writes P+1 .. P+1+k
            if (k < 0) return false;
            if (conf_tau_s >= 0) {
                int kk = 0;
                while (kk < k && 1.0f / (1.0f + std::exp(-r.sp_conf[kk])) >= conf_tau_s) ++kk;
                k = kk;
            }
            const bool refeed = r.dirty && ((r.P + 1) & 1);
            int dr[kVerify] = {0};
            for (int i = 0; i < k; ++i) dr[i] = r.sp_ids[i];
            bool is_lk = false;
            if (lk_max > 0 && r.lk_cool > 0) --r.lk_cool;
            else if (lk_max > 0) {
                // the sequence is hist + t (t = the token for P+1, not in hist yet); find the latest earlier q with
                // seq[q-G .. q] == seq[n-G .. n] (G+1 tokens, the last being t), then copy seq[q+1 ..]
                const int n = (int) hist.size();
                const int t = r.t;
                auto sq = [&](int i) { return i == n ? t : hist[(size_t) i]; };
                int cap = std::min(lk_max, kVerify - 1 - (refeed ? 1 : 0));
                if (r.max_new > 0) cap = std::min(cap, r.max_new - r.gen);
                cap = std::min(cap, ctxv - (r.P + 3));
                if (cap >= 2 && n > lk_g + 1) {
                    for (int q2 = n - 1; q2 >= lk_g; --q2) {
                        if (hist[(size_t) q2] != t) continue;
                        int m = 1;
                        while (m < 64 && q2 - m >= 0 && hist[(size_t) (q2 - m)] == sq(n - m)) ++m;
                        if (m < lk_g + 1) continue;
                        const int L = std::min({cap, m - lk_g + 1, n - q2});
                        if (L < 2) break;   // the latest match copies too little: not worth a pass
                        if (lk_agree && sq(q2 + 1) != r.sp_ids[0]) break;
                        for (int i = 0; i < L; ++i) dr[i] = sq(q2 + 1 + i);
                        k = L;
                        is_lk = true;
                        break;
                    }
                }
            }
            vt.clear();
            if (refeed) vt.push_back(r.prev);
            vt.push_back(r.t);
            for (int i = 0; i < k; ++i) vt.push_back(dr[i]);
            const int r0 = refeed ? 1 : 0, p0 = refeed ? r.P : r.P + 1;
            if (!pass_v(vt.data(), (int) vt.size(), p0)) { r.finish = "error"; return false; }
            int acc = 0, tn = -1;
            for (; acc < k; ++acc) {
                sample_dist(vlog.data() + (size_t) (r0 + acc) * nv, nv, r.temp, r.tk, r.top_p, r.min_p, dist);
                if (!spec_accept(dist, dr[acc], rng, tn)) break;
            }
            if (acc == k) tn = sample_p(vlog.data() + (size_t) (r0 + k) * nv, nv, r.temp, r.tk, r.top_p, r.min_p, rng);
            for (int i = 0; i <= acc && rowdump_f; ++i) rowdump(r.id, r.P + 1 + i, vt[(size_t) (r0 + i)], vlog.data() + (size_t) (r0 + i) * nv);
            ++r.rq_pass; r.rq_acc += acc;
            if (is_lk) {
                ++r.lk_pass; r.lk_dr += k; r.lk_acc += acc;
                if (2 * acc < k) { r.lk_cool = r.lk_back; r.lk_back = std::min(64, r.lk_back * 2); }
                else r.lk_back = 1;
            }
            int spos[kVerify + 1];
            for (int i = 0; i <= acc; ++i) spos[i] = r.P + 1 + i;
            int drafts[kVerify];
            for (int i = 0; i < k; ++i) drafts[i] = dr[i];
            if (!dsp->step(vh.data() + (size_t) r0 * 3 * n_embd, spos, acc + 1, tn, r.sp_ids, r.sp_conf)) {
                ferr = "DSpark step: " + dsp->last_error(); r.finish = "error"; return false;
            }
            hist.push_back(r.t); pen_record(r.t);
            bool more = true;
            for (int i = 0; i < acc && more; ++i) { hist.push_back(drafts[i]); pen_record(drafts[i]); more = emit_tok(r, drafts[i]); }
            r.prev = acc == 0 ? r.t : drafts[acc - 1];
            r.dirty = acc < k;
            r.P += 1 + acc;
            r.t = tn;
            tier.rebalance_tick();
            if (!more) return false;
            return emit_tok(r, r.t);
        };
#endif
        // ---- one plain decode step of the live request r (no drafts): feed the token emitted last, sample and emit the next
        auto run_step = [&](Req & r) -> bool {
            if (r.stop_req) { r.finish = "cancel"; return false; }
            if (r.pos + 1 >= ctxv) return false;
            if (!step(r.ptok, r.pos) || !dense.logits(0, 1, r.lg.data())) { r.finish = "error"; return false; }
            rowdump(r.id, r.pos, r.ptok, r.lg.data());
            hist.push_back(r.ptok); pen_record(r.ptok);
            ++r.pos;
            tier.rebalance_tick();
            r.ptok = sample_p(r.lg.data(), nv, r.temp, r.tk, r.top_p, r.min_p, rng);
            return emit_tok(r, r.ptok);
        };
        // ---- one batch window (dslots > 1; docs/BATCHING.md): a row of every resident running request through every layer in
        // ONE pass (Ds41Dense::begin_rows: the dense weights, the shared expert and every routed expert two rows share are read
        // once for all of them), each row sampled with its request's own sampler.  No drafts in a window - here a pass is bound
        // by the expert bytes it streams, and a draft row would cost its experts for ~0.4 tokens - but a DSpark request's
        // drafter window is kept fed (STRATA_SERVE_WIN_DSPARK=0: not), so it drafts again the moment it runs alone.  A request
        // that ends in a window is made live first (end_req reads the live state).
        std::vector<Ds41Dense::Row> w_rows;
        std::vector<Req *> w_req;
        std::vector<int> w_want;
        auto run_window = [&]() {
            if (live_req) park_req(*live_req);
            w_rows.clear(); w_req.clear(); w_want.clear();
            auto finish = [&](Req & q, const char * how) {
                if (how) q.finish = how;
                if (!make_live(q)) q.finish = "error";
                end_req(q, false);
            };
            for (auto & q : reqs) {
                if (!q.active || q.live || q.dslot < 0) continue;
                if (q.stop_req) { finish(q, "cancel"); continue; }
                if (q.spec ? ctxv - (q.P + 3) < 0 : q.pos + 1 >= ctxv) { finish(q, nullptr); continue; }
                if ((int) w_rows.size() + 2 > Ds41Dense::kMaxRows) break;   // a window holds kMaxRows rows
                if (q.spec) {
                    if (q.dirty && ((q.P + 1) & 1)) w_rows.push_back(Ds41Dense::Row { q.dslot, q.P, q.prev });   // the refeed (run_pass)
                    w_rows.push_back(Ds41Dense::Row { q.dslot, q.P + 1, q.t });
                } else {
                    w_rows.push_back(Ds41Dense::Row { q.dslot, q.pos, q.ptok });
                }
                w_want.push_back((int) w_rows.size() - 1);
                w_req.push_back(&q);
            }
            if (w_req.empty()) return;
            const double t0 = now_ms();
            if (!pass_v(nullptr, (int) w_rows.size(), 0, w_rows.data())) {
                std::fprintf(stderr, "serve: batch window failed: %s\n", ferr.c_str());
                for (Req * q : w_req) finish(*q, "error");
                return;
            }
            const double t1 = now_ms();
            win_ms += t1 - t0;
            ++win_n;
            win_rows += (long) w_rows.size();
            for (size_t i = 0; i < w_req.size(); ++i) {
                Req & q = *w_req[i];
                const int row = w_want[i];
                const float * lg = vlog.data() + (size_t) row * nv;
                q.stamp = ++run_clock;
                std::swap(g_pen, q.pen);   // the request's own penalty window (pen_record / sample_p)
                bool more;
                if (q.spec) {
                    rowdump(q.id, q.P + 1, q.t, lg);
                    const int tn = sample_p(lg, nv, q.temp, q.tk, q.top_p, q.min_p, q.rng);
#if defined(DS41_HAVE_DSPARK)
                    static const bool win_dspark = !(std::getenv("STRATA_SERVE_WIN_DSPARK") && std::getenv("STRATA_SERVE_WIN_DSPARK")[0] == '0');
                    if (dsp && win_dspark && !q.dsp_stale) {   // this position into its drafter window, drafts after tn
                        const double td0 = now_ms();
                        const int spos1 = q.P + 1;
                        bool ok = true;
                        if (q.dsp_saved) ok = dsp->window_in(q.dsp_win, q.dsp_filled); else dsp->reset();
                        ok = ok && dsp->step(vh.data() + (size_t) row * 3 * n_embd, &spos1, 1, tn, q.sp_ids, q.sp_conf);
                        ok = ok && dsp->window_out(q.dsp_win, q.dsp_filled);
                        q.dsp_saved = ok;
                        if (!ok) q.dsp_stale = true;
                        win_dsp_ms += now_ms() - td0;
                    } else {
                        q.dsp_stale = true;
                    }
#endif
                    q.hist.push_back(q.t); pen_record(q.t);
                    q.prev = q.t; q.dirty = false; q.P += 1; q.t = tn;
                    ++q.rq_pass;
                    more = emit_tok(q, tn);
                } else {
                    rowdump(q.id, q.pos, q.ptok, lg);
                    q.hist.push_back(q.ptok); pen_record(q.ptok);
                    ++q.pos;
                    q.ptok = sample_p(lg, nv, q.temp, q.tk, q.top_p, q.min_p, q.rng);
                    more = emit_tok(q, q.ptok);
                }
                std::swap(g_pen, q.pen);
                ++q.w_pass;
                if (!more) finish(q, nullptr);
            }
            tier.rebalance_tick();
            const double tw = now_ms() - t0;
            win_post_ms += tw - (t1 - t0);
            if (tw > 1000.0) {   // a stall: graph rebuilds (a layout change, a full graph context), a cache rebalance, the drafter
                ++win_slow;
                win_slow_max = std::max(win_slow_max, tw);
                std::fprintf(stderr, "serve: slow batch window: %.0f ms (pass %.0f, after it %.0f; %d rows)\n", tw, t1 - t0, tw - (t1 - t0), (int) w_rows.size());
            }
        };
        // ---- a request: GEN <max_new> [k=v ..] <ids csv> (solo) or BGEN <slot> <max_new> [k=v ..] <ids csv> (a batch slot);
        // GENI / BGENI carry an image-embedding file before the csv (GENI <max_new> [k=v ..] <embeddings file> <ids csv>: the
        // file holds one strata-vision record per image, in prompt order; the engine lands them on the image_token runs).
        // The prompt is read (reusing what the live conversation and the parks hold), the first token is emitted; a slot
        // request then gets its DONE and BADM lines and decodes in turns with the other slots, a solo request runs on alone.
        auto admit = [&](const std::string & line, bool bg) {
            const bool geni = line.rfind("GENI ", 0) == 0 || line.rfind("BGENI ", 0) == 0;
            std::istringstream ss(line);
            std::string w, csv, emb_file;
            int slot = -1, max_new = 0, tk = 0;
            float temp = a.temp, top_p = 1.0f, min_p = 0.0f;
            float req_rb = cfg_route_bias, req_rbf = cfg_route_bias_file;
            int req_rprot = cfg_rprot, req_rpool = cfg_rpool;
            PenaltyState pen;   // the request's own penalties (none unless its keys say so)
            uint64_t seed = 0;
            bool seeded = false;
            ss >> w;
            if (bg) ss >> slot;
            ss >> max_new;
            while (ss >> w) {
                const size_t eq = w.find('=');
                if (eq == std::string::npos) {   // the two positionals: GENI's embeddings file then the csv (GEN: just the csv)
                    if (geni && emb_file.empty()) emb_file = w; else csv = w;
                    continue;
                }
                const std::string k = w.substr(0, eq), v = w.substr(eq + 1);
                if (k == "temperature") temp = (float) std::atof(v.c_str());
                else if (k == "top_p") top_p = (float) std::atof(v.c_str());
                else if (k == "top_k") tk = std::atoi(v.c_str());
                else if (k == "min_p") min_p = (float) std::atof(v.c_str());
                else if (k == "penalty_repeat") pen.repeat = (float) std::atof(v.c_str());
                else if (k == "penalty_freq") pen.freq = (float) std::atof(v.c_str());
                else if (k == "penalty_present") pen.present = (float) std::atof(v.c_str());
                else if (k == "penalty_last_n") pen.last_n = std::max(0, std::atoi(v.c_str()));
                else if (k == "seed") { seed = (uint64_t) std::atoll(v.c_str()); seeded = true; }
                else if (k == "route_bias") req_rb = std::max(0.0f, (float) std::atof(v.c_str()));
                else if (k == "route_bias_file") req_rbf = std::max(0.0f, (float) std::atof(v.c_str()));
                else if (k == "route_protect") req_rprot = std::max(0, std::atoi(v.c_str()));
                else if (k == "route_pool") req_rpool = std::max(0, std::atoi(v.c_str()));
            }
            if (bg && (slot < 0 || slot >= n_slots || reqs[(size_t) slot].active)) {
                std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", n_slots);
                std::fflush(stdout);
                return;
            }
            if (!bg && reqs.back().active) { std::printf("ERR GEN: a request is still running\n"); std::fflush(stdout); return; }
            std::vector<int> ids = csv.empty() ? std::vector<int>() : parse_csv(csv);
            if (ids.empty()) { std::printf("ERR empty prompt\n"); std::fflush(stdout); return; }
            if ((int64_t) ids.size() + std::max(0, max_new) > ctxv) {
                std::printf("ERR prompt (%zu) + max tokens (%d) exceed the context (%d)\n", ids.size(), max_new, ctxv);
                std::fflush(stdout);
                return;
            }
            Req & r = bg ? reqs[(size_t) slot] : reqs.back();
            r = Req();
            r.slot = bg ? slot : -1;
            r.max_new = max_new;
            r.temp = temp; r.top_p = top_p; r.min_p = min_p; r.tk = tk;
            r.route_bias = req_rb; r.route_bias_file = req_rbf; r.rprot = req_rprot; r.rpool = req_rpool;
            // the device: an active request's conversation goes to its Req first; a finished one stays for the reuse rules
            if (live_req) park_req(*live_req);
            if (dslots > 1) {   // the prompt is read into a slot no running request holds (theirs stay resident)
                int d = free_slot();
                if (d < 0) {    // every slot holds a running request: the least recently run one goes to a host park
                    Req * victim = nullptr;
                    for (auto & q : reqs) if (q.active && !q.live && q.dslot >= 0 && (!victim || q.stamp < victim->stamp)) victim = &q;
                    if (!victim || !evict_req(*victim)) { std::printf("ERR no conversation slot free\n"); std::fflush(stdout); return; }
                    d = dense.slot();
                }
                if (!switch_slot(d)) { std::printf("ERR %s\n", ferr.c_str()); std::fflush(stdout); return; }
            }
            g_pen = pen;
            if (seeded) rng.seed(seed);
            ensure_snap_rings();
            dense.clear_vision();
            if (previous_vision) {
                hist.clear();
                snap_len = -1;
                live_end = 0;
                previous_vision = false;
            }
            // a changed bias is applied again just before the decode: a repeated prompt (regenerate) runs no prompt pass,
            // and the passes are what turn the decode bias on - without this it decoded with the previous request's
            bool bias_changed = false;
            if (req_rb != a.route_bias || req_rbf != a.route_bias_file) {
                a.route_bias = req_rb;
                a.route_bias_file = req_rbf;
                bias_changed = true;
                std::fprintf(stderr, "serve: decode route bias %.2f, file penalty %.2f%s\n", req_rb, req_rbf,
                             (req_rb == 0.0f && req_rbf == 0.0f) ? " (exact routing)" : "");
            }
            if ((req_rprot != cur_rprot || req_rpool != cur_rpool) && dense.set_route_select(req_rprot, req_rpool)) {
                cur_rprot = req_rprot;
                cur_rpool = req_rpool;
                std::fprintf(stderr, "serve: smart routing: exact top %d always kept, picks inside the exact top %d%s\n",
                             req_rprot, req_rpool, (req_rprot == 0 && req_rpool == 0) ? " (off: the flat bias)" : "");
            }
            // vision: resolve this request's image rows onto its image_token runs and hand them to the dense layers
            std::vector<float> vis_rows;
            std::vector<int64_t> vis_pos;
            const bool is_vision = geni;
            if (is_vision) {
                if (!dense.vision()) { std::printf("ERR image input requires --vision\n"); std::fflush(stdout); return; }
                std::string verr;
                std::vector<std::vector<float>> recs;
                std::vector<int64_t> counts;
                const int64_t ne = dense.geom().n_embd;
                if (!read_sve1(emb_file, recs, counts, ne, (int64_t) ids.size(), verr)
                    || !map_vision(ids, a.image_token, recs, counts, vis_rows, vis_pos, verr)) {
                    std::printf("ERR %s\n", verr.c_str()); std::fflush(stdout); return;
                }
                if (vis_rows.empty()) { std::printf("ERR GENI prompt has no image tokens\n"); std::fflush(stdout); return; }
                // an image prompt prefills fresh: every image span is the same run of image_token, so a token-only reuse
                // test would "continue" a different image (same ids, different embeddings).  Empty hist / no snapshot / no park.
                park_conv(live_end);   // the text conversation goes to a slot (its rows are about to be overwritten)
                hist.clear();
                snap_len = -1;
                live_end = 0;
                dense.set_vision(vis_rows.data(), vis_pos.data(), (int) vis_pos.size());
                previous_vision = true;
            }
            size_t lcp = 0;
            while (lcp < hist.size() && lcp < ids.size() && hist[lcp] == ids[lcp]) ++lcp;
            size_t div = lcp;          // the longest start this prompt shares with any conversation the engine knows
            for (const auto & o : reqs) {   // ... the requests running in the other slots included
                if (!o.active || o.live) continue;
                size_t l2 = 0;
                while (l2 < o.hist.size() && l2 < ids.size() && o.hist[l2] == ids[l2]) ++l2;
                div = std::max(div, l2);
            }
            bool unparked = false;
            int reuse_override = -1;   // a resume below a park's end (the tail rules): the position to restart at
            size_t lp_show = 0;        // how far this request continues the PARKED conversation it took (logged)
            if (!parks.empty() && !is_vision) {   // does a parked conversation serve this request better than the live one?
                // what the live conversation gives it (the rules below, evaluated early): its whole history, the snapshot's
                // prompt, or - the tail rule - an even restart no further than rollback_max() below the snapshot's end
                const int rb = dense.rollback_max();
                size_t live_reuse = 0;
                if (!hist.empty() && lcp == hist.size() && lcp < ids.size()) live_reuse = (size_t) live_end;
                else if (snap_len > 0 && lcp >= (size_t) snap_len) live_reuse = (size_t) snap_len;
                else if (tail_resume && snap_len > 0 && (lcp & ~(size_t) 1) >= 16 && snap_len - (int) (lcp & ~(size_t) 1) <= rb)
                    live_reuse = lcp & ~(size_t) 1;
                // STRATA_SERVE_PARK_PARTIAL (default on): a request that continues all but the last few tokens of a parked
                // conversation (a harness that re-renders its tail, or cancels and sends again with a note appended) resumes
                // it too.  The window ring and the ratio-2 compressor state are NOT position-indexed (see Ds41Dense::
                // snapshot): the resume restores the rings kept at the park's END or at its prompt end and starts at an EVEN
                // position no further than rollback_max() below that point, whose older rows are still in them.
                static const bool park_partial = !(std::getenv("STRATA_SERVE_PARK_PARTIAL") && std::getenv("STRATA_SERVE_PARK_PARTIAL")[0] == '0');
                // STRATA_SERVE_PARK_NO_REGEN=1 (tests only): a parked copy of the IDENTICAL prompt is not served from the park (it would
                // answer from its saved logits instead of computing the request, which a numerics test needs)
                static const bool park_no_regen = std::getenv("STRATA_SERVE_PARK_NO_REGEN") && std::getenv("STRATA_SERVE_PARK_NO_REGEN")[0] == '1';
                enum { M_NONE, M_END, M_END_TAIL, M_SNAP, M_SNAP_TAIL };
                int best = -1, best_mode = M_NONE;
                size_t best_reuse = live_reuse, best_lp = 0;
                for (size_t i = 0; i < parks.size(); ++i) {
                    const ParkEntry & e = parks[i];
                    const std::vector<int> & ph = e.hist;
                    size_t lp = 0;
                    while (lp < ph.size() && lp < ids.size() && ph[lp] == ids[lp]) ++lp;
                    div = std::max(div, lp);
                    const size_t E = ph.size(), Sn = (size_t) e.snap_len;
                    size_t reuse = 0;
                    int mode = M_NONE;
                    if (lp == E) {
                        if (lp < ids.size()) { reuse = E; mode = M_END; }                        // continues it: restart at its END
                        else if (Sn == E && !park_no_regen && !e.logits.empty()) { reuse = E; mode = M_END; }   // the same prompt again: its logits are saved
                        else if (E >= 18) { reuse = (E - 2) & ~(size_t) 1; mode = M_END_TAIL; }   // identical to a decoded one: just below its end
                    } else if (lp > Sn) {                                                       // the prompt, and part of what followed
                        const size_t R = lp & ~(size_t) 1;
                        if (park_partial && R >= 16 && E - R <= (size_t) rb) { reuse = R; mode = M_END_TAIL; }
                        else { reuse = Sn; mode = M_SNAP; }
                    } else if (lp == Sn) { reuse = Sn; mode = M_SNAP; }
                    else if (park_partial && (lp & ~(size_t) 1) >= 16 && Sn - (lp & ~(size_t) 1) <= (size_t) rb) { reuse = lp & ~(size_t) 1; mode = M_SNAP_TAIL; }
                    if (reuse > best_reuse) { best = (int) i; best_reuse = reuse; best_lp = lp; best_mode = mode; }
                }
                lp_show = best_lp;
                if (best >= 0) {
                    const bool is_pfx = parks[(size_t) best].prefix;
                    ParkEntry e;
                    if (is_pfx) {   // a shared prefix is served by copy and stays
                        e = parks[(size_t) best];
                        parks[(size_t) best].stamp = ++park_clock;
                        ++parks[(size_t) best].hits;
                    } else {
                        e = std::move(parks[(size_t) best]);
                        parks.erase(parks.begin() + best);
                    }
                    park_conv(live_end);   // the live conversation first: swapping in overwrites its rows
                    const bool at_prompt = best_mode == M_SNAP || best_mode == M_SNAP_TAIL;
                    bool ok = dense.park_in(e.p);
                    if (ok && at_prompt && !e.prompt_rings.empty()) ok = dense.rings_in(e.prompt_rings, e.snap_len);
                    if (!ok) {
                        hist.clear();
                        snap_len = -1;
                        live_end = 0;
                        std::fprintf(stderr, "serve: unpark failed - reading in full\n");
                    } else {
                        hist = std::move(e.hist);
                        snap_logits = std::move(e.logits);
                        snap_seed = std::move(e.seed);
                        snap_seed_n = e.seed_n;
                        // the live snapshot rings from here: the prompt-end ones for a resume at the prompt, else the END's (a
                        // continuation takes a new snapshot at its own prompt end anyway)
                        if (at_prompt && !e.prompt_rings.empty()) snap_rings = std::move(e.prompt_rings); else dense.rings_out(snap_rings);
                        snap_rings_dev = true;
                        unparked = true;   // the live rings are another conversation's: resume through restore()
#if defined(DS41_HAVE_DSPARK)
                        dsp_next = -1;     // DSpark's window holds the other conversation's states, unless the park kept this one's
#endif
                        const char * what = is_pfx ? "shared prefix" : "conversation";
                        if (best_mode == M_END) {
                            snap_len = (int) hist.size();   // the generic rules below: continue at (or regenerate from) the END
                            live_end = snap_len;
                            lcp = best_lp;
#if defined(DS41_HAVE_DSPARK)
                            if (dsp && !e.dsp_win.empty() && dsp->window_in(e.dsp_win, e.dsp_filled)) dsp_next = snap_len;
#endif
                            std::fprintf(stderr, "serve: unparked the %zu-token %s%s (%zu left parked)\n", hist.size(), what,
                                         is_pfx ? (" (hit " + std::to_string(parks[(size_t) best].hits) + ")").c_str() : "", parks.size());
                        } else if (best_mode == M_SNAP) {
                            snap_len = e.snap_len;          // the generic rules: continue at the prompt end / the same prompt again
                            live_end = (int) hist.size();
                            lcp = best_lp;
                            std::fprintf(stderr, "serve: unparked the %zu-token %s at its prompt end %d (%zu left parked)\n", hist.size(), what,
                                         e.snap_len, parks.size());
                        } else {   // M_END_TAIL / M_SNAP_TAIL: the rings of that point, an even restart below it, re-fed from there
                            const size_t R = best_reuse;
                            const bool ring_ok = dense.restore();
                            std::fprintf(stderr, "serve: unparked the %zu-token %s at lcp %zu (tail: re-feeds from %zu below %s, %zu left parked)\n",
                                         hist.size(), what, best_lp, R, best_mode == M_END_TAIL ? "its end" : "its prompt end", parks.size());
                            if (ring_ok) { hist.resize(R); reuse_override = (int) R; lcp = R; }
                            else { hist.clear(); lcp = 0; }
                            snap_len = -1;   // the tail under that point is re-fed, so no snapshot survives this
                            live_end = (int) hist.size();
                        }
                    }
                }
            }
            int reused = 0;
            bool regen = false;   // the snapshot's own prompt again: nothing new to feed
            if (reuse_override >= 0) reused = reuse_override;   // a tail resume through the park: positions [0, R) are the parked one's
            else if (!unparked && !hist.empty() && lcp == hist.size() && lcp < ids.size()) {
                reused = live_end;   // continue the live conversation (after an open compressor group: the group's first token again)
                if ((size_t) live_end < hist.size()) hist.resize((size_t) live_end);
            } else if (snap_len > 0 && lcp >= (size_t) snap_len && (size_t) snap_len == ids.size() && !snap_logits.empty() && (ensure_snap_rings(), dense.restore())) {
                hist.resize((size_t) snap_len);   // the same prompt again (regenerate): its end state + saved logits
                reused = snap_len;
                last_logits = snap_logits;
                regen = true;
            } else if (snap_len > 0 && lcp >= (size_t) snap_len && (size_t) snap_len < ids.size() && (ensure_snap_rings(), dense.restore())) {
                hist.resize((size_t) snap_len);
                reused = snap_len;
            } else if (tail_resume && snap_len > 0 && !is_vision && lcp < (size_t) snap_len &&
                       snap_len - (int) (lcp & ~(size_t) 1) <= dense.rollback_max() && (int) (lcp & ~(size_t) 1) >= 16 &&
                       (ensure_snap_rings(), dense.restore())) {
                // STRATA_SERVE_TAIL_RESUME (default on): the prompt differs from the snapshot's within its last few tokens
                // (the harness cancelled a request and sent it again with something appended: the last token of
                // "</tool_result>" merges with the next characters and the generation header goes after the new text).
                // The snapshot ring holds RAW = SWA + 16 rows, so the window of a query up to rollback_max() positions
                // below the snapshot is still there; the position-indexed caches below R are the same tokens' rows.  R is
                // made even so no ratio-2 compressor group is open at R.  2026-10-22: four such resends re-read 77-111K
                // tokens each (15.7 min of the day) because rule 3 needs lcp >= snap_len.
                const int r_even = (int) (lcp & ~(size_t) 1);
                std::fprintf(stderr, "serve: tail resume: the prompt leaves the snapshot's %d tokens %d before its end - restart at %d\n",
                             snap_len, snap_len - (int) lcp, r_even);
                hist.resize((size_t) r_even);
                reused = r_even;
                snap_len = -1;   // rows from r_even on are rewritten by this request: the old snapshot is stale
            } else {
                // an unrelated request: park the conversation it would overwrite (see STRATA_SERVE_PARK)
                park_conv(live_end);
                hist.clear();
                snap_len = -1;
            }
            if (reused > 0) dense.prime_history(ids.data(), std::max(0, reused - 16), reused);   // the n-gram look-back of the first re-fed tokens
            std::fprintf(stderr, "serve: reuse lcp=%zu park=%zu lp=%zu hist=%zu snap=%d prompt=%zu div=%zu -> reused=%d\n", lcp,
                         parks.size(), lp_show, hist.size(), snap_len, ids.size(), div, reused);
            if (reused > 0) { std::printf("RESUME %d\n", reused); std::fflush(stdout); }
            const int total = (int) ids.size();
            // a long prompt runs layer-major without the VRAM cache (its pass / chunk buffers need that VRAM) and with
            // the prompt's routing (no decode bias).  A short one (<= step_max new tokens: a chat / agent turn's tool
            // result) runs on the decode side instead - small dense passes, the cache and the decode biases: the
            // layer-major path streams nearly every expert of every layer whatever the batch size (~18 s for 40 tokens)
            const int delta = total - reused;
            int sp_tail = 0;   // DSpark: the last new prompt tokens, fed by pass_v (seeds the window)
#if defined(DS41_HAVE_DSPARK)
            if (dsp_path) {
                if (!(reused > 0 && reused == dsp_next) && dsp) dsp->reset();
                dsp_next = -1;
                sp_tail = std::min(128, delta);
            }
#endif
            int dh = delta - sp_tail;   // the part the passes / batch feed
            bool by_step = defer_vram && dh >= 1 && dh <= step_max;
            // STRATA_SERVE_BATCH_TAIL (default on, =0 the old way): a turn that runs as a layer-major batch anyway takes
            // its DSpark tail INTO the batch and the seed rows are read off the batch's state (bcap_vh) - the tail as
            // 16-token exact passes was half of every turn (s11 2026-10-15: +273 tokens 33.5 s = batch 17.1 + tail 15.8).
            // DSpark drafts are verified, so only the seed's numerics (batch vs pass kernels) change, never the output.
            static const bool batch_tail = !(std::getenv("STRATA_SERVE_BATCH_TAIL") && std::getenv("STRATA_SERVE_BATCH_TAIL")[0] == '0');
            int bcap = 0;
            if (batch_tail && sp_tail > 0 && !by_step && dh >= 2 && PB > 1) { bcap = sp_tail; dh = delta; }
            // STRATA_SERVE_BATCH_MIN (tokens; default 128, 0 = off): a turn of at least this many new tokens runs as a
            // layer-major batch even when the passes would take it (every turn up to 128 + STEP_MAX = 224 tokens did:
            // the DSpark tail + the step path).  The batch's floor fell with the routed-only reads.  s11, 512K, prompt s
            // (+ 32-token decode) passes / batch: +30 5.0 (1.7) / 6.0 (2.0), +60 6.6 (1.7) / 7.0 (2.0), +90 9.7 (1.0) /
            // 7.7 (1.7), +150 17.6 (1.4) / 9.8 (1.6).  Decode right after a batch starts colder (the VRAM cache is
            // re-seeded), hence the margin above the ~80-token crossover.
            static const int batch_min = std::getenv("STRATA_SERVE_BATCH_MIN") ? std::atoi(std::getenv("STRATA_SERVE_BATCH_MIN")) : 128;
            if (batch_min > 0 && batch_tail && sp_tail > 0 && bcap == 0 && delta >= batch_min && PB > 1 && delta >= 2) {
                bcap = sp_tail; dh = delta; by_step = false;
            }
            static const int dump_tail = std::getenv("STRATA_SERVE_DUMP_TAIL") ? std::atoi(std::getenv("STRATA_SERVE_DUMP_TAIL")) : 0;
            // the shared-prefix park (see ParkEntry): this read is split at `pfx`, the state there goes to a park slot
            int pfx = -1;
            if (prefix_min > 0 && !by_step && !is_vision && dh >= 2 && PB > 1 && dump_tail == 0) {
                const int d = (int) (div & ~(size_t) 1);
                bool have = false;
                for (const auto & e : parks) if (e.prefix && (int) e.hist.size() == d) have = true;
                if (!have && d - reused >= prefix_min && d >= 16 && reused + dh - d >= 2) pfx = d;
            }
            if (by_step && !cache_open) {
                if (!open_cache()) { std::printf("ERR VRAM seed failed: %s\n", err.c_str()); std::fflush(stdout); return; }
                cache_open = true;
            }
            // exact passes (STRATA_SERVE_PASS_BIAS=0): the decode-side prompt tokens route with the model's own
            // selection (no route bias / file penalty) - only the batch shape differs from the exact batch path
            const bool exact_pass = by_step && !pass_bias;
            if (exact_pass && bias_set) {
                bias_now = a.bias_prompt > 0;
                apply_bias();
            }
            // STRATA_SERVE_KEEP_CACHE=1 (s6, opt-in until measured): a batch turn of at most STRATA_SERVE_KEEP_MAX new
            // tokens (default 4096) gives back only the VRAM its batch needs (the calibrated need(B) below) - the cache's
            // lowest-ranked tail - instead of the whole cache, so the hot experts serve the batch from VRAM (no PCIe) and
            // the decode after it skips the re-seed (regrow_cache refills just the tail).  The prompt still routes exactly.
            // s6 A/B: it never engaged with the default --vram-lru (the LRU reorders slots, so "the tail" was not the
            // coldest experts); since 2026-10-23 shrink_cache_for gathers the LRU-hottest experts below the shrink point
            // first (Ds41MoeTier::shrink_cache_for, ExpertCache::swap_slots), so it works with the LRU: no re-seed after
            // the batch (p50 2 s, p90 6 s per read on 2026-10-22) and the decode after it starts with the warm set.
            bool kept = false;
            if (Ds41MoeTier::keep_cache_mode() && !by_step && dh >= 2 && !(defer_vram && cache_open))
                std::fprintf(stderr, "serve: keep-cache: not applicable (deferred VRAM %d, cache open %d)\n", (int) defer_vram,
                             (int) cache_open);
            if (!by_step && defer_vram && cache_open && dh >= 2) {
                static const int keep_max = std::getenv("STRATA_SERVE_KEEP_MAX") ? std::atoi(std::getenv("STRATA_SERVE_KEEP_MAX")) : 4096;
                // continuations only (2026-10-24, keepcache-ab2): the kept LRU set is the live conversation's; a new
                // conversation's first decode ran 18% slower on it than on the fresh ranked seed, an append +5% faster
                if (Ds41MoeTier::keep_cache_mode() && dh <= keep_max && !mc.cpu_only && reused > 0) {
                    // the same fit as pb_cap's below, incl. the long-context attention term (S * total * 10 B)
                    const double need = (4.3 * 1073741824.0 + 0.95 * 1048576.0 * (double) std::min(dh, PB)) +
                                        (double) S * (double) total * 10.0 + std::max(1.0, a.vram_margin_gib) * 1073741824.0;
                    std::string kerr;
                    const int64_t g = tier.shrink_cache_for((int64_t) need, kerr);
                    if (g >= 0) {
                        kept = true;
                        std::fprintf(stderr, "serve: cache kept for a %d-token batch (%lld tail slots out, %lld experts stay)\n",
                                     dh, (long long) g, (long long) tier.resident());
                    } else if (!kerr.empty()) {
                        std::fprintf(stderr, "serve: keep-cache: %s - releasing the cache\n", kerr.c_str());
                    }
                }
                if (!kept) {
                    tier.release_cache();
                    cache_open = false;
                    info_vram();
#if defined(DS41_HAVE_DSPARK)
                    // STRATA_SERVE_DSPARK_EVICT=N (s6, opt-in until measured; 0 = off): a batch of >= N new tokens also
                    // frees DSpark's ~2.9 GB of weights for its duration - the prompt batch is sized from free VRAM
                    // (s6: 8731 columns before DSpark loads, 5434 after), so a long prompt takes fewer expert sweeps.
                    // open_cache() re-creates it for the tail (~1-4 s); the window is re-seeded by the tail as always.
                    static const int dsp_evict = std::getenv("STRATA_SERVE_DSPARK_EVICT") ? std::atoi(std::getenv("STRATA_SERVE_DSPARK_EVICT")) : 0;
                    if (dsp && dsp_evict > 0 && dh >= dsp_evict) {
                        dsp.reset();
                        std::fprintf(stderr, "serve: DSpark weights freed for a %d-token batch\n", dh);
                    }
#endif
                }
                bias_now = a.bias_prompt > 0;
                apply_bias();
            }
            tier.rebalance_begin_prompt();
            tier.reset_stats();
            if (const char* rl = std::getenv("STRATA_ROUTE_LOG")) {   // a turn marker for the route log (see ds41_moe.cpp)
                if (FILE* f = std::fopen(rl, "a")) { std::fprintf(f, "# turn %d new %d %s\n", n_req, dh, by_step ? "passes" : "batch"); std::fclose(f); }
            }
            const double tp0 = now_ms();
            auto pp = [&](int done) {
                const double ms = now_ms() - tp0;
                std::printf("PP %d %d %.0f %.1f\n", reused + done, total, ms, 1000.0 * done / std::max(ms, 1e-9));
                std::fflush(stdout);
            };
            auto feed_steps = [&]() -> bool {   // the short-turn path: passes of up to pass_max tokens (a lone token: a step)
                int last_rows = 1;
                feed_done = 0;
                feed_cancelled = false;
                for (int i = 0; i < dh; ) {
                    const int m = std::min(pass_max, dh - i);
                    if (!(m == 1 ? step(ids[(size_t) (reused + i)], reused + i) : pass_m(ids.data() + reused + i, m, reused + i))) return false;
                    if (!capture_tail(delta, i, m, 0)) return false;   // this pass's rows, before the next pass replaces them
                    const int before = i;
                    i += m;
                    last_rows = m;
                    if (i < dh && stop_pending()) { feed_done = i; feed_cancelled = true; ferr = "cancelled"; return false; }
                    if (i / 64 != before / 64 && i < dh) pp(i);
                }
                feed_done = dh;
                if (!dense.logits(last_rows - 1, 1, last_logits.data())) { ferr = dense.last_error(); return false; }
                pp(dh);
                return true;
            };
            tail_keep = dump_tail > 0 ? std::min(dump_tail, delta) : 0;
            tail_logits.clear();
            // VRAM-aware batch width for this turn (feed() halves it again if even this does not fit): the batch's
            // state columns + graph are ~0.48 MiB a column, and the card is shared with whatever else the desktop
            // runs.  This is what lets a big --prompt-batch stay OPT-IN: the engine takes what is actually free.
            auto size_pb = [&]() {   // (a lambda: run again if a kept cache is released after all, below)
                pb_cap = PB;
                if (!by_step && dh >= 2 && PB > 1 && !mc.cpu_only) {
                    size_t frb = 0, totb = 0;
                    ggml_backend_dev_memory(ggml_backend_get_device(be), &frb, &totb);
                    // The layer-major batch's own VRAM, CALIBRATED on four traced points (STRATA_VRAM_TRACE=1, s13
                    // run9/run13/run14, a 4 GiB foreign process): free 9.28 GiB with B=8192 failed, free 9.26 with
                    // B=5210 passed, free 5.92 with B=2411 failed, a free card with B=8192 passed - all four fit
                    //     need(B) ~= 4.3 GiB + 0.95 MiB * B
                    // 4.3 GiB fixed = the tier's two 1 GiB stage halves + a ~2.19 GiB graph buffer that is the SAME
                    // SIZE at B=2411 and B=8192 (so it is not the piece buffers; identity unconfirmed - see the
                    // residuals in V41_START_HERE).  0.95 MiB a column = columns 0.32 + graph 0.17 + part buffers 0.41,
                    // measured directly.  Per-tensor arithmetic undercounted the tier 2x and let a 2.19 GiB allocation
                    // fail while cudaMemGetInfo still said 6.83 GiB free.  A free card (>= 13 GiB) still gets its full PB.
                    // run20 (2026-10-16) fixes the GRAPH term: the pass-graph buffer is B-INDEPENDENT - the B=16384 leg
                    // OOM'd on "allocating 1424.31 MiB", the very size this comment records for 8192, so "graph 0.17 MiB
                    // a column" is a mis-derivation (1424/8192) and the decomposition is really ~5.7 GiB fixed + 0.78 a
                    // column.  Same line, same fit - but it matters: a wide batch's per-column STATE is what starves the
                    // fixed graph buffer (16384 = 5.14 GiB of state, 1.03 GiB short), not the graph itself.
                    // the fit above was calibrated with the default 1 GiB margin: a smaller --vram-margin (more decode
                    // slots) must not widen the batch past it (a 22K prompt at --vram-margin 0.5 took B=11914 and its
                    // 2.28 GiB graph buffer did not fit, 2026-10-18)
                    // long prompts (2026-10-21, the 524K needle): a dense sub-batch's attention graph grows with the
                    // context (index scores / top-k over every position for S rows): the 4150-column batch at 524K wanted
                    // a 4.66 GiB graph buffer, 2.47 GiB over the ~2.19 GiB the fixed term assumes, and the request failed
                    // ("graph allocation failed").  ~10 B per position per sub-batch row, so the fixed term grows by
                    // S * total * 10 B (0.64 GiB at 131K, 5 GiB at 1M); the fit shrinks the batch instead of failing
                    const double ctx_term = (double) S * (double) total * 10.0;
                    const double room = (double) frb - std::max(1.0, a.vram_margin_gib) * 1073741824.0 - 4.3 * 1073741824.0 - ctx_term;
                    const int fit = (int) std::max<int64_t>(2, (int64_t) (room / (0.95 * 1048576.0)));
                    pb_cap = std::min(PB, std::max(SB, fit));
                    if (vram_trace) std::fprintf(stderr, "vram trace: cap free %.2f GiB room %.2f GiB fit %d -> pb_cap %d\n",
                                                 frb / 1073741824.0, room / 1073741824.0, fit, pb_cap);
                    if (pb_cap < PB) std::fprintf(stderr, "serve: VRAM %.2f GiB free -> prompt batch %d (of %d)\n",
                                                  frb / 1073741824.0, pb_cap, PB);
                }
            };
            size_pb();
            if (kept && pb_cap < std::min(dh, PB)) {
                // keep-cache: the shrink was meant to leave room for the whole batch (the same fit); if the card
                // says otherwise the kept cache would cost extra expert sweeps - release it, as without keep-cache
                std::fprintf(stderr, "serve: keep-cache: the kept cache leaves room for %d of %d columns - releasing it after all\n",
                             pb_cap, std::min(dh, PB));
                tier.release_cache();
                cache_open = false;
                kept = false;
                info_vram();
                size_pb();
            }
            const double tph0 = now_ms();
            const double fd0[3] = { fd_dense, fd_tier, fd_logits };   // this request's batch split (stderr)
            bool ok = true;
            int read_upto = reused;   // tokens of the prompt read completely (a cancelled read keeps the conversation up to here)
            if (total == reused || dh == 0) {
            } else if (by_step) {
                ok = feed_steps();
            } else if (pfx > reused && pfx < reused + dh) {
                // the shared prefix as a batch of its own: the rings, rows, logits and DSpark seed rows at its end go to a park
                // slot, then the rest of the prompt (the park is taken once per engine run; the next request that starts the
                // same way restarts at pfx instead of reading it)
                const int n1 = pfx - reused;
                bcap_n = std::min(128, n1); bcap_got = 0;
                bcap_vh.assign((size_t) bcap_n * 3 * n_embd, 0.0f);
                const double tx0 = now_ms();
                ok = feed(ids.data() + reused, n1, reused, pp, nullptr);
                if (!ok) read_upto = reused + feed_done;
                if (ok) {
                    Ds41Dense::ParkedPtr p = dense.snapshot(pfx) ? dense.park_out((int64_t) pfx + 64) : nullptr;
                    snap_rings_dev = false;
                    if (p) {
                        ParkEntry e;
                        e.p = std::move(p);
                        e.hist.assign(ids.begin(), ids.begin() + pfx);
                        e.snap_len = pfx;
                        e.logits = last_logits;   // feed() left the prefix's last row there
                        if (bcap_got > 0) {
                            e.seed.assign(bcap_vh.begin(), bcap_vh.begin() + (size_t) bcap_got * 3 * n_embd);
                            e.seed_n = bcap_got;
                        }
                        e.prefix = true;
                        e.stamp = ++park_clock;
                        const double need = (double) Ds41Dense::parked_bytes(e.p) + (double) e.seed.size() * 4;
                        size_t n_pfx = 0;
                        for (const auto & x : parks) n_pfx += x.prefix ? 1 : 0;
                        while (n_pfx >= prefix_slots && n_pfx > 0) {   // the least recently used prefix makes room
                            auto pick = parks.end();
                            for (auto it = parks.begin(); it != parks.end(); ++it)
                                if (it->prefix && (pick == parks.end() || it->stamp < pick->stamp)) pick = it;
                            std::fprintf(stderr, "serve: prefix slots full - dropped the %zu-token shared prefix (%d hits)\n", pick->hist.size(), pick->hits);
                            prefix_file_drop(*pick);
                            parks.erase(pick);
                            --n_pfx;
                        }
                        if (need <= park_cap) {
                            evict_parks(need);
                            prefix_file_save(e);
                            parks.push_back(std::move(e));
                            std::fprintf(stderr, "serve: shared prefix of %d tokens parked (%.0f MiB, %.2f s to read; %zu parked, %.0f MiB)\n", pfx,
                                         need / 1048576.0, (now_ms() - tx0) / 1000.0, parks.size(), parks_bytes() / 1048576.0);
                        }
                    }
                    bcap_n = bcap; bcap_got = 0;
                    if (bcap > 0) bcap_vh.assign((size_t) bcap * 3 * n_embd, 0.0f);
                    auto pp2 = [&](int done) { pp(n1 + done); };
                    ok = feed(ids.data() + pfx, reused + dh - pfx, pfx, pp2, nullptr);
                    if (!ok) read_upto = pfx + feed_done;
                }
            } else {
                bcap_n = bcap; bcap_got = 0;
                if (bcap > 0) bcap_vh.assign((size_t) bcap * 3 * n_embd, 0.0f);
                ok = feed(ids.data() + reused, dh, reused, pp, nullptr);
                if (!ok) read_upto = reused + feed_done;
            }
            if (!ok && by_step) read_upto = reused + feed_done;
            bcap_n = 0;
            if (kept) {   // the batch is done: the tail slots back (exactly the evicted experts), decode biases on
                std::string kerr;
                dense.release_pass();   // the batch's pass buffers first (open_cache does the same before a seed)
                if (tier.regrow_cache(kerr) < 0) {
                    std::fprintf(stderr, "serve: keep-cache regrow: %s - re-seeding\n", kerr.c_str());
                    tier.release_cache();
                    cache_open = false;
                } else {
                    bias_now = true;
                    apply_bias();
                }
            }
            double ph_feed = now_ms() - tph0, ph_open = 0, ph_tail = 0;   // the request's prompt phases (stderr)
            int sp_t = -1;   // DSpark: the first decode token (sampled from the last prompt row), its drafts in sp_ids
            int sp_ids[5] = {0}, sp_tmp[5];
            float sp_conf[5] = {0};
#if defined(DS41_HAVE_DSPARK)
            if (ok && sp_tail > 0) {   // the tail through pass_v (chunks of <= 16), the cache open; routing as the passes
                const double to0 = now_ms();
                if (defer_vram && !cache_open) {
                    if (!open_cache()) { ferr = "VRAM seed failed: " + err; ok = false; }
                    else cache_open = true;
                }
                ph_open = now_ms() - to0;
                if (ok) { bias_now = pass_bias; apply_bias(); }
                const double tt0 = now_ms();
                std::vector<int> spos;
                if (bcap > 0) {   // the seed rows came off the batch (bcap_vh, its last bcap_got tokens): DSpark steps only
                    const int c0s = total - bcap_got;
                    for (int i0 = c0s; ok && i0 < total; i0 += 16) {
                        const int m = std::min(16, total - i0);
                        spos.resize((size_t) m);
                        for (int t = 0; t < m; ++t) spos[(size_t) t] = i0 + t;
                        const bool last = i0 + m == total;
                        if (last) sp_t = sample_p(last_logits.data(), nv, temp, tk, top_p, min_p, rng);   // the batch's last row
                        if (i0 == c0s) cur_seed_n = 0;
                        cur_seed.resize((size_t) (cur_seed_n + m) * 3 * n_embd);
                        std::copy_n(bcap_vh.data() + (size_t) (i0 - c0s) * 3 * n_embd, (size_t) m * 3 * n_embd,
                                    cur_seed.data() + (size_t) cur_seed_n * 3 * n_embd);
                        cur_seed_n += m;
                        if (!dsp->step(bcap_vh.data() + (size_t) (i0 - c0s) * 3 * n_embd, spos.data(), m, last ? sp_t : -1,
                                       last ? sp_ids : sp_tmp, sp_conf)) {
                            ferr = "DSpark step: " + dsp->last_error(); ok = false; break;
                        }
                    }
                    if (bcap_got < 1) { ferr = "DSpark batch seed captured nothing"; ok = false; }
                }
                for (int i0 = total - sp_tail; ok && bcap == 0 && i0 < total; i0 += 16) {
                    const int m = std::min(16, total - i0);
                    if (!pass_v(ids.data() + i0, m, i0)) { ok = false; break; }
                    spos.resize((size_t) m);
                    for (int t = 0; t < m; ++t) spos[(size_t) t] = i0 + t;
                    const bool last = i0 + m == total;
                    if (last) {
                        std::copy(vlog.end() - nv, vlog.end(), last_logits.begin());
                        sp_t = sample_p(last_logits.data(), nv, temp, tk, top_p, min_p, rng);
                    }
                    if (i0 == total - sp_tail) cur_seed_n = 0;
                    cur_seed.resize((size_t) (cur_seed_n + m) * 3 * n_embd);
                    std::copy_n(vh.data(), (size_t) m * 3 * n_embd, cur_seed.data() + (size_t) cur_seed_n * 3 * n_embd);
                    cur_seed_n += m;
                    if (!dsp->step(vh.data(), spos.data(), m, last ? sp_t : -1, last ? sp_ids : sp_tmp, sp_conf)) {
                        ferr = "DSpark step: " + dsp->last_error(); ok = false; break;
                    }
                }
                if (ok) { bias_now = true; apply_bias(); }
                ph_tail = now_ms() - tt0;
                pp(delta);
            } else if (ok && regen && dsp && snap_seed_n > 0) {   // a repeated prompt: re-seed DSpark from the snapshot's rows
                const double tt0 = now_ms();
                dsp->reset();
                std::vector<int> spos;
                const int s0 = total - snap_seed_n;
                for (int i0 = s0; i0 < total; i0 += 16) {
                    const int m = std::min(16, total - i0);
                    spos.resize((size_t) m);
                    for (int t = 0; t < m; ++t) spos[(size_t) t] = i0 + t;
                    const bool last = i0 + m == total;
                    if (last) sp_t = sample_p(last_logits.data(), nv, temp, tk, top_p, min_p, rng);
                    if (!dsp->step(snap_seed.data() + (size_t) (i0 - s0) * 3 * n_embd, spos.data(), m, last ? sp_t : -1,
                                   last ? sp_ids : sp_tmp, sp_conf)) {
                        std::fprintf(stderr, "serve: DSpark re-seed: %s (decoding without drafts)\n", dsp->last_error().c_str());
                        dsp->reset();
                        sp_t = -1;
                        break;
                    }
                }
                ph_tail = now_ms() - tt0;
                std::fprintf(stderr, "serve: repeated prompt - DSpark re-seeded from %d saved rows (%.2f s)\n", snap_seed_n, ph_tail / 1000.0);
            }
#endif
            dense.prof_report("turn");
            std::fprintf(stderr, "serve: prompt phases: %d new = %d %s (%.2f s: dense %.2f, experts %.2f, logits %.2f) + DSpark tail %d%s "
                         "(cache open %.2f s, seed %.2f s)\n", delta, dh, by_step ? "passes" : "batch", ph_feed / 1000.0,
                         (fd_dense - fd0[0]) / 1000.0, (fd_tier - fd0[1]) / 1000.0, (fd_logits - fd0[2]) / 1000.0,
                         bcap > 0 ? bcap_got : sp_tail, bcap > 0 ? " from the batch" : " as passes", ph_open / 1000.0, ph_tail / 1000.0);
            if (tail_keep > 0 && ok) {   // the quality scorer's rows: req%03d.tail.f32, [tail_keep x n_vocab], last prompt row last
                if (const char * dd = std::getenv("STRATA_SERVE_DUMP")) {
                    char nb[40];
                    std::snprintf(nb, sizeof(nb), "/req%03d.tail.f32", n_req);
                    std::ofstream f(std::string(dd) + nb, std::ios::binary);
                    f.write((const char *) tail_logits.data(), (std::streamsize) (tail_logits.size() * 4));
                }
            }
            tail_keep = 0;
            if (!ok && feed_cancelled) {
                // the read was cancelled (STOP): keep the conversation up to read_upto (the rings there, a snapshot, no
                // logits) and end the request as cancelled; the resend continues from there
                feed_cancelled = false;
                if (read_upto >= 16 && dense.snapshot(read_upto)) {
                    hist.assign(ids.begin(), ids.begin() + read_upto);
                    snap_len = read_upto;
                    live_end = read_upto;
                    snap_logits.clear();   // none at this point: the same prompt again re-feeds from here
                    snap_seed.clear();
                    snap_seed_n = 0;
                    dense.rings_out(snap_rings);
                    snap_rings_dev = true;
                } else {
                    hist.clear();
                    snap_len = -1;
                    live_end = 0;
                }
#if defined(DS41_HAVE_DSPARK)
                dsp_next = -1;
#endif
                cur_seed_n = 0;
                if (defer_vram && !cache_open && open_cache()) cache_open = true;   // decode next: the cache back
                const double pms = now_ms() - tp0;
                std::fprintf(stderr, "serve: prompt read cancelled after %d of %d tokens (%.2f s): the conversation stays loaded to %d\n",
                             read_upto, total, pms / 1000.0, (int) hist.size());
                std::printf("DONE 0 %d %.1f 0.0 cancel 0 0 %d 0 0 0 0 0 %d\n", total, pms, reused, read_upto - reused);
                if (bg) std::printf("BADM %d 0\n", slot);
                std::fflush(stdout);
                return;
            }
            if (!ok) {
                std::printf("ERR prefill failed: %s\n", ferr.c_str());
                std::fflush(stdout);
                hist.clear();
                snap_len = -1;
                live_end = 0;
                return;
            }
            hist = ids;
            live_end = total;
            // no snapshot for an image prompt: a text request with the same ids (all image_token) would restore its embeddings
            if (total > reused && !is_vision && dense.snapshot(total)) {
                snap_len = total;
                snap_logits = last_logits;
                snap_seed.swap(cur_seed);   // this prompt's DSpark tail rows (0 rows when DSpark is off / no tail ran)
                snap_seed_n = cur_seed_n;
                dense.rings_out(snap_rings);
                snap_rings_dev = true;
            }
            cur_seed_n = 0;
            if (const char * dd = std::getenv("STRATA_SERVE_DUMP")) {   // exactness checks: request N's last-row logits
                char nb[32];
                std::snprintf(nb, sizeof(nb), "/req%03d.f32", n_req);
                std::ofstream f(std::string(dd) + nb, std::ios::binary);
                f.write((const char *) last_logits.data(), (std::streamsize) (last_logits.size() * 4));
            }
            r.id = n_req;
            ++n_req;
            const double prompt_ms = now_ms() - tp0;
            if (exact_pass && cache_open) {   // decode: the biases back on, from the VRAM set as it is now
                bias_now = true;
                apply_bias();
            }
            const Ds41MoeStats pst = tier.stats();
            std::fprintf(stderr, "serve: prompt %d tokens (%d reused, %s) in %.2f s = %.1f tok/s (experts %lld hit / %lld cpu / "
                                 "%lld pcie / %lld file)\n", total, reused, by_step ? (exact_pass ? "exact passes" : "passes") : "batch", prompt_ms / 1000.0,
                         1000.0 * (total - reused) / std::max(prompt_ms, 1e-9), (long long) pst.hits, (long long) pst.cpu,
                         (long long) pst.pcie, (long long) pst.file_tier);
            if (defer_vram && !cache_open) {   // decode next: the VRAM cache opens into what the prompt freed
                if (!open_cache()) {
                    std::printf("ERR VRAM seed failed: %s\n", err.c_str());
                    std::fflush(stdout);
                    return;
                }
                cache_open = true;
                std::fprintf(stderr, "serve: VRAM %.2f GiB free -> %lld experts in VRAM (seed %.2f s)\n", seed_free,
                             (long long) tier.resident(), seed_s);
            }
            tier.rebalance_plan();
            if (bias_changed) {   // decode: this request's route bias + file penalty (see bias_changed)
                bias_now = true;
                apply_bias();
            }
            tier.reset_stats();
            dense.route_meter(true);
            // ---- the request is admitted: its first token, then its decode state (the slots take turns from here)
            r.total = total;
            r.reused = reused;
            r.prompt_ms = prompt_ms;
            r.td0 = now_ms();
            r.active = true;
            r.live = true;
            live_req = &r;
            r.spec = sp_t >= 0;
            r.P = total - 1;
            r.t = sp_t;
            r.prev = ids.back();
            for (int i = 0; i < 5; ++i) { r.sp_ids[i] = sp_ids[i]; r.sp_conf[i] = sp_conf[i]; }
            r.pos = total;
            bool more;
            if (r.spec) more = emit_tok(r, r.t);
            else {
                r.lg = last_logits;
                r.ptok = sample_p(r.lg.data(), nv, temp, tk, top_p, min_p, rng);
                more = emit_tok(r, r.ptok);
            }
            if (bg) {   // the admission's own figures, then whether the slot goes on (the server reads BT / BDONE from here)
                const bool cont = more;
                std::printf("DONE %d %d %.1f 0.0 %s 0 0 %d %lld %lld 0 %lld 0 %d\n", r.gen, total, prompt_ms, more ? "length" : r.finish, reused,
                            (long long) pst.hits, (long long) pst.lookups(), (long long) pst.file_tier, total - reused);
                std::printf("BADM %d %d\n", slot, cont ? 1 : 0);
                std::fflush(stdout);
                if (!cont) end_req(r, true);
                return;
            }
            if (!more) end_req(r, false);
        };
        // ---- the loop: commands as they come; between them, one pass of the next active request (round robin,
        // slice_passes passes each).  Idle = wait for a line.
        std::string line;
        size_t rr = 0;
        for (;;) {
            bool have = false;
            if (any_active()) have = try_line(line);
            else { if (!next_line(line)) break; have = true; }
            if (have) {
                if (line == "QUIT") break;
                if (line.empty()) continue;
                if (line == "STOP") {   // the solo request's
                    if (reqs.back().active) reqs.back().stop_req = true;
                    continue;
                }
                if (line.rfind("BSTOP ", 0) == 0) {
                    const int s = std::atoi(line.c_str() + 6);
                    if (s >= 0 && s < n_slots && reqs[(size_t) s].active) reqs[(size_t) s].stop_req = true;
                    continue;
                }
                if (line.rfind("BYIELD ", 0) == 0) continue;   // a prompt read is one batch here: it never gives way (the server copes)
                const bool bg = line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0;
                if (!bg && line.rfind("GEN ", 0) != 0 && line.rfind("GENI ", 0) != 0) { std::printf("ERR unsupported command\n"); std::fflush(stdout); continue; }
                admit(line, bg);
                continue;
            }
            if (at_eof() && !any_active()) break;
            if (dslots > 1) {
                // batch windows: 2+ running requests and every one of them resident -> one pass with a row of each; with a
                // host-parked one among them the turns below, as with one slot
                int n_res = 0, n_host = 0;
                for (const auto & q : reqs) if (q.active) { if (q.live || q.dslot >= 0) ++n_res; else ++n_host; }
                if (n_res >= 2 && n_host == 0) { run_window(); continue; }
            }
            // the next active request: the current one until its slice is used up, then round robin
            Req * cur = (rr < reqs.size() && reqs[rr].active && reqs[rr].slice < slice_passes) ? &reqs[rr] : nullptr;
            if (!cur) {
                if (rr < reqs.size()) reqs[rr].slice = 0;
                for (size_t k2 = 1; k2 <= reqs.size() && !cur; ++k2) {
                    const size_t i = (rr + k2) % reqs.size();
                    if (reqs[i].active) { cur = &reqs[i]; rr = i; }
                }
                if (!cur) continue;
                cur->slice = 0;
            }
            Req & r = *cur;
            if (!make_live(r)) { r.finish = "error"; end_req(r, false); continue; }
            ++r.slice;
#if defined(DS41_HAVE_DSPARK)
            const bool more = r.spec ? run_pass(r) : run_step(r);
#else
            const bool more = run_step(r);
#endif
            if (!more) end_req(r, false);
        }
        tier.close();
        // the stdin reader thread (detached) sits in getline holding cin's lock: a normal exit's stream teardown waits
        // on it until the client closes the pipe - leave without the static destructors
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(0);
    }

    // ---- one-shot: prompt (+ perplexity), then generate
    if (prompt.empty()) { std::fprintf(stderr, "ds41_generate: empty prompt\n"); return 2; }
    tier.reset_stats();
    const double tp0 = now_ms();
    std::vector<float> all;
    const int dm_tail = dump_main ? std::min<int>(128, (int) prompt.size() - 1) : 0;   // DSpark's window, as steps
    bool spec = dsp_path != nullptr;
#if !defined(DS41_HAVE_DSPARK)
    if (spec) { std::fprintf(stderr, "ds41_generate: STRATA_DSPARK: built without ds41_dspark\n"); return 2; }
#endif
    const int spec_depth = std::getenv("STRATA_DSPARK_DEPTH") ? std::atoi(std::getenv("STRATA_DSPARK_DEPTH")) : 4;
    if (spec && (!defer_vram || S < 17 || dump_main || spec_depth < 0 || spec_depth > 5 || std::getenv("STRATA_FORCE_IDS"))) {
        std::fprintf(stderr, "ds41_generate: STRATA_DSPARK needs the GPU layer-major prompt path (S >= 17), depth 1..5, and no "
                             "STRATA_DUMP_MAIN / STRATA_FORCE_IDS\n");
        return 2;
    }
    if (spec && a.temp > 0) std::fprintf(stderr, "note: STRATA_DSPARK decodes greedily (temp ignored)\n");
    // the DSpark window needs per-position main_hidden, which the layer-major prompt does not produce: the last
    // min(128, L-1) prompt tokens run through pass_v (chunks of <= 16) after the VRAM cache opens
    const int sp_tail = spec ? std::min<int>(128, (int) prompt.size() - 1) : 0;
    const bool ok = feed(prompt.data(), (int) prompt.size() - dm_tail - sp_tail, 0, [&](int done) {
        if (!a.quiet) std::fprintf(stderr, "\rprompt %d/%zu", done, prompt.size());
    }, a.ppl || !a.dump_logits.empty() ? &all : nullptr);
    if (!ok) { std::fprintf(stderr, "\nds41_generate: prefill: %s\n", ferr.c_str()); return 1; }
    if (dm_tail > 0 && defer_vram && !open_cache()) { std::fprintf(stderr, "ds41_generate: deferred seed: %s\n", err.c_str()); return 1; }
    for (int i = (int) prompt.size() - dm_tail; i < (int) prompt.size(); ++i)
        if (!step(prompt[(size_t) i], i) || !dense.logits(0, 1, last_logits.data())) {
            std::fprintf(stderr, "ds41_generate: dump-main prompt step: %s\n", ferr.c_str());
            return 1;
        }
#if defined(DS41_HAVE_DSPARK)
    int sp_bonus = -1;   // the trunk's token for position L (argmax of the last prompt row) and DSpark's first drafts
    int sp_ids[5] = {0}, sp_tmp[5];
    float sp_conf[5];
    if (spec) {
        if (!open_cache()) { std::fprintf(stderr, "ds41_generate: deferred seed: %s\n", err.c_str()); return 1; }
        bias_now = false;   // the tail is PROMPT: it routes like the rest of the prompt (no decode biases), as plain decode
        apply_bias();
        std::vector<int> spos;
        for (int i0 = (int) prompt.size() - sp_tail; i0 < (int) prompt.size(); i0 += 16) {
            const int m = std::min<int>(16, (int) prompt.size() - i0);
            if (!pass_v(prompt.data() + i0, m, i0)) { std::fprintf(stderr, "ds41_generate: DSpark seed: %s\n", ferr.c_str()); return 1; }
            spos.resize((size_t) m);
            for (int t = 0; t < m; ++t) spos[(size_t) t] = i0 + t;
            const bool last = i0 + m == (int) prompt.size();
            if (last) {
                std::copy(vlog.end() - nv, vlog.end(), last_logits.begin());
                sp_bonus = (int) (std::max_element(last_logits.begin(), last_logits.end()) - last_logits.begin());
            }
            if (!dsp->step(vh.data(), spos.data(), m, last ? sp_bonus : -1, last ? sp_ids : sp_tmp, sp_conf)) {
                std::fprintf(stderr, "ds41_generate: DSpark seed step: %s\n", dsp->last_error().c_str());
                return 1;
            }
        }
        bias_now = true;    // decode: route bias + file penalty on (open_cache's state)
        apply_bias();
    }
#endif
    const double prompt_ms = now_ms() - tp0;
    if (defer_vram && dm_tail == 0 && !spec) {   // the prompt is done: its buffers go back, the VRAM expert cache opens into the free space
        if (!open_cache()) { std::fprintf(stderr, "ds41_generate: deferred seed: %s\n", err.c_str()); return 1; }
        std::fprintf(stderr, "\nVRAM after the prompt: %.2f GiB free -> %.2f GiB of expert slots: %lld experts in VRAM (%.1f s)",
                     seed_free, seed_gib, (long long) tier.resident(), seed_s);
    }
    const Ds41MoeStats pst = tier.stats();
    std::fprintf(stderr, "\nprompt: %zu tokens in %.2f s = %.1f tok/s (experts %lld hit / %lld cpu / %lld pcie / %lld file%s)\n",
                 prompt.size(), prompt_ms / 1000.0, 1000.0 * prompt.size() / prompt_ms, (long long) pst.hits,
                 (long long) pst.cpu, (long long) pst.pcie, (long long) pst.file_tier,
                 pst.skipped > 0 ? (", skipped " + std::to_string(pst.skipped)).c_str() : "");
    std::fprintf(stderr, "prompt split: dense %.1f s, experts %.1f s (file wait %.1f s), logits %.1f s\n", fd_dense / 1000.0,
                 fd_tier / 1000.0, pst.file_ms / 1000.0, fd_logits / 1000.0);
    if (a.ppl) {
        double nll = 0;
        int n = 0, top1 = 0;
        for (size_t t = 0; t + 1 < prompt.size(); ++t) {
            const float * lg = all.data() + t * (size_t) nv;
            nll -= log_softmax_at(lg, nv, prompt[t + 1]);
            top1 += (int) (std::max_element(lg, lg + nv) - lg) == prompt[t + 1];
            ++n;
        }
        std::printf("PPL %.4f  mean NLL %.4f over %d tokens, top-1 %.3f\n", std::exp(nll / std::max(1, n)), nll / std::max(1, n), n,
                    (double) top1 / std::max(1, n));
    }
    if (!a.dump_logits.empty()) {
        std::ofstream f(a.dump_logits, std::ios::binary);
        f.write((const char *) all.data(), (std::streamsize) (all.size() * 4));
    }
    if (!a.dump_last.empty() && a.dense_bench.empty()) {   // the last row's logits - the one row enc-skip still computes exactly
        std::ofstream f(a.dump_last, std::ios::binary);
        f.write((const char *) last_logits.data(), (std::streamsize) (last_logits.size() * 4));
    }
    if (!a.dense_bench.empty()) {
        // --dense-bench R:P (see Args): the dense half alone, P passes of R rows (the shape of DSpark's verify passes), the
        // tier skipped, routed sums zero.  Tokens cycle through the prompt, positions continue after it.
        int R = 2, P = 50;
        if (std::sscanf(a.dense_bench.c_str(), "%d:%d", &R, &P) < 1 || R < 1 || R > 16 || P < 1) {
            std::fprintf(stderr, "ds41_generate: --dense-bench wants R:P with 1 <= R <= 16\n");
            return 2;
        }
        if (defer_vram && !open_cache()) { std::fprintf(stderr, "ds41_generate: deferred seed: %s\n", err.c_str()); return 1; }
        tier_skip = true;
        {
            std::vector<float> z((size_t) R * n_embd, 0.0f);
            if (!dense.upload_routed(z.data(), R)) std::fprintf(stderr, "dense-bench: note: could not zero the routed sums (%s)\n", dense.last_error().c_str());
        }
        for (double & v : pv_ms) v = 0;
        for (double & v : pvd_ms) v = 0;
        pv_n = 0; pv_tok = 0;
        std::ofstream dump;
        if (!a.dump_last.empty()) dump.open(a.dump_last, std::ios::binary);
        std::vector<int> bt((size_t) R);
        std::vector<double> ms((size_t) P);
        const int pos_b = (int) prompt.size();
        for (int p = 0; p < P; ++p) {
            for (int i = 0; i < R; ++i) bt[(size_t) i] = prompt[(size_t) ((p * R + i) % (int) prompt.size())];
            const int pos = pos_b + p * R;
            if (pos + R > (int) a.ctx) { std::fprintf(stderr, "dense-bench: the context is full at pass %d\n", p); P = p; break; }
            const double t0 = now_ms();
            if (!pass_v(bt.data(), R, pos)) { std::fprintf(stderr, "ds41_generate: dense-bench pass %d: %s\n", p, ferr.c_str()); return 1; }
            ms[(size_t) p] = now_ms() - t0;
            if (dump.is_open()) dump.write((const char *) vlog.data(), (std::streamsize) ((size_t) R * nv * 4));
        }
        std::vector<double> sorted(ms.begin(), ms.begin() + P);
        std::sort(sorted.begin(), sorted.end());
        double mean = 0;
        const int skip = std::min(P - 1, 2);   // the first passes build / warm the graphs
        for (int p = skip; p < P; ++p) mean += ms[(size_t) p];
        mean /= std::max(1, P - skip);
        std::fprintf(stderr, "dense-bench: %d rows x %d passes: %.2f ms/pass mean (after %d warm-up), min %.2f, median %.2f, max %.2f; "
                     "per layer %.3f ms\n", R, P, mean, skip, sorted.front(), sorted[(size_t) P / 2], sorted.back(), mean / n_layer);
        std::fprintf(stderr, "dense-bench split (all %ld passes): begin %.2f, predict %.2f, attn_router %.2f, finish %.2f, logits %.2f ms/pass; "
                     "attn_router %.1f us/layer, finish %.1f us/layer, predict %.1f us/layer\n", pv_n, pvd_ms[0] / pv_n, pvd_ms[1] / pv_n,
                     pvd_ms[2] / pv_n, pvd_ms[3] / pv_n, pv_ms[3] / pv_n, 1000.0 * pvd_ms[2] / pv_n / n_layer, 1000.0 * pvd_ms[3] / pv_n / n_layer,
                     1000.0 * pvd_ms[1] / pv_n / n_layer);
        dense.prof_report("dense-bench");
        tier.close();
        return 0;
    }
    tier.reset_stats();
    // STRATA_PROFILE_DECODE=1: nsys --capture-range=cudaProfilerApi captures exactly the decode
    static const bool prof_decode = std::getenv("STRATA_PROFILE_DECODE") != nullptr;
    if (prof_decode) Ds41MoeTier::profiler(true);
    const double td0 = now_ms();
    std::vector<float> lg = last_logits;
    int pos = (int) prompt.size(), gen = 0;
    // STRATA_FORCE_IDS=<file>: decode this token sequence (one id per line - a previous run's stdout) in place of the
    // sampled tokens, so kernel A/Bs whose variants round differently still do the same work (greedy decoding flips on
    // near-ties, and a different text means different experts and a different VRAM hit rate)
    std::vector<int> forced;
    if (const char * fp = std::getenv("STRATA_FORCE_IDS")) {   // lines that are not a bare id (a memguard banner) skipped
        std::ifstream f(fp);
        for (std::string line; std::getline(f, line);) {
            char * end = nullptr;
            const long t = std::strtol(line.c_str(), &end, 10);
            if (end != line.c_str() && *end == '\0') forced.push_back((int) t);
        }
        std::fprintf(stderr, "STRATA_FORCE_IDS: %zu forced tokens from %s\n", forced.size(), fp);
    }
#if defined(DS41_HAVE_DSPARK)
    // ---- DSpark: P = the last fed position, t = the trunk's token for P+1 (not fed yet), d = drafts for P+2..
    // A pass verifies [t, d0..d(k-1)] at P+1..P+1+k; g_i = the argmax of row P+1+i (the trunk's token for P+2+i);
    // a = #leading d_i == g_i; d0..d(a-1) + t' = g_a are emitted, DSpark appends the a+1 accepted rows and drafts
    // after t'.  Every cache is position-indexed (rejected rows are overwritten when re-fed; the window ring is sized
    // for it), except the ratio-2 compressor's 2-slot state ring: a pass writes its LAST 2 tokens' states, so after a
    // rejection a pass that starts at an odd position re-feeds the last accepted token (its group's state then comes
    // from inside the pass).
    // STRATA_ROUTE_TRACE=<file> (measurement only): the verify passes' routing + acceptance, for
    // tools/v41/sim/route_sim.py (session 13: in-pass draft pruning and the VRAM replacement policy, replayed offline)
    if (const char * rt = std::getenv("STRATA_ROUTE_TRACE")) tier.trace_open(rt);
    long sp_pass = 0, sp_acc = 0, sp_refeed = 0, sp_full = 0;
    const bool sp_noacc = std::getenv("STRATA_DSPARK_NOACC") != nullptr;
    double sp_vms = 0, sp_dms = 0;
    if (spec) {
        int P = (int) prompt.size() - 1, t = sp_bonus, prev = prompt.back();
        bool dirty = false;   // the last pass rejected something
        std::vector<int> vt;
        auto emit = [&](int tok) -> bool {   // false = stop here
            std::printf("%d\n", tok);
            std::fflush(stdout);
            ++gen;
            return !(std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end() || tok == (int) G.eos || gen >= a.n_predict);
        };
        // STRATA_DSPARK_CONF=<tau>: verify only the leading drafts whose confidence-head probability (sigmoid of
        // conf) is >= tau - a draft that will be rejected still costs its experts' PCIe / NVMe bytes in the verify
        // pass, while every emitted token is still the trunk's own (lossless: only the verify width changes).
        // STRATA_DSPARK_CONFLOG=<file>: per pass "k acc verify_ms p0 p1 p2 p3 p4" (the offline threshold screen).
        static const float conf_tau = std::getenv("STRATA_DSPARK_CONF") ? (float) std::atof(std::getenv("STRATA_DSPARK_CONF")) : -1.0f;
        static FILE * conf_log = std::getenv("STRATA_DSPARK_CONFLOG") ? std::fopen(std::getenv("STRATA_DSPARK_CONFLOG"), "w") : nullptr;
        auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        while (emit(t)) {
            int k = std::min(spec_depth, a.n_predict - gen);   // drafts past the budget are never emitted
            if (conf_tau >= 0) {
                int kk = 0;
                while (kk < k && sig(sp_conf[kk]) >= conf_tau) ++kk;
                k = kk;
            }
            const bool refeed = dirty && ((P + 1) & 1);
            vt.clear();
            if (refeed) vt.push_back(prev);
            vt.push_back(t);
            for (int i = 0; i < k; ++i) vt.push_back(sp_ids[i]);
            const int r0 = refeed ? 1 : 0, p0 = refeed ? P : P + 1;
            const double tv = now_ms();
            if (!pass_v(vt.data(), (int) vt.size(), p0)) { std::fprintf(stderr, "ds41_generate: verify: %s\n", ferr.c_str()); return 1; }
            sp_vms += now_ms() - tv;
            // STRATA_DSPARK_DUMP=<file> (correctness tap): every verify row as {pos, m, row, refeed, nv floats} - row
            // `pos` predicts the token at pos+1, so it lines up with row `pos` of a --dump-logits prefill of the stream
            static FILE * sp_dump = std::getenv("STRATA_DSPARK_DUMP") ? std::fopen(std::getenv("STRATA_DSPARK_DUMP"), "wb") : nullptr;
            if (sp_dump) {
                for (int i = 0; i < (int) vt.size(); ++i) {
                    const int32_t hdr[4] = { p0 + i, (int32_t) vt.size(), i, refeed ? 1 : 0 };
                    std::fwrite(hdr, 4, 4, sp_dump);
                    std::fwrite(vlog.data() + (size_t) i * nv, 4, (size_t) nv, sp_dump);
                }
                std::fflush(sp_dump);
            }
            int g[6];
            for (int i = 0; i <= k; ++i) {
                const float * row = vlog.data() + (size_t) (r0 + i) * nv;
                g[i] = (int) (std::max_element(row, row + nv) - row);
            }
            int acc = 0;
            while (acc < k && sp_ids[acc] == g[acc]) ++acc;
            if (sp_noacc) acc = 0;   // bisect tap: every draft rejected (the stream must stay plain greedy)
            ++sp_pass; sp_acc += acc; sp_refeed += refeed; sp_full += acc == k;
            if (conf_log) {
                // acceptance of EVERY draft against the trunk rows this pass computed (beyond k: unknown, -1)
                std::fprintf(conf_log, "%d %d %.2f", k, acc, now_ms() - tv);
                for (int i = 0; i < spec_depth; ++i) std::fprintf(conf_log, " %.4f", sig(sp_conf[i]));
                std::fprintf(conf_log, "\n");
                std::fflush(conf_log);
            }
            {   // STRATA_ROUTE_TRACE: 'P' {m, k, acc, refeed, verify ms, conf p0..p4}
                float pm[10] = { (float) vt.size(), (float) k, (float) acc, (float) refeed, (float) (now_ms() - tv) };
                for (int i = 0; i < 5; ++i) pm[5 + i] = i < spec_depth ? sig(sp_conf[i]) : 0.0f;
                tier.trace_mark(pm, 10);
            }
            int spos[6];
            for (int i = 0; i <= acc; ++i) spos[i] = P + 1 + i;
            const int drafts[5] = { sp_ids[0], sp_ids[1], sp_ids[2], sp_ids[3], sp_ids[4] };
            const double td = now_ms();
            if (!dsp->step(vh.data() + (size_t) r0 * 3 * n_embd, spos, acc + 1, g[acc], sp_ids, sp_conf)) {
                std::fprintf(stderr, "ds41_generate: DSpark step: %s\n", dsp->last_error().c_str());
                return 1;
            }
            sp_dms += now_ms() - td;
            bool more = true;
            for (int i = 0; i < acc && more; ++i) more = emit(drafts[i]);
            if (!more) break;
            prev = acc == 0 ? t : drafts[acc - 1];
            dirty = acc < k;
            P += 1 + acc;
            t = g[acc];
        }
    }
#endif
    for (; !spec && gen < a.n_predict; ++gen) {
        int tok = sample_p(lg.data(), nv, a.temp, 0, 1.0f, 0.0f, rng);
        if (gen < (int) forced.size()) {
            // STRATA_DEC_MARGIN=1 (gate tap, with STRATA_FORCE_IDS): per step the argmax, its logit and the forced
            // token's logit - a forced stream that leaves the greedy path at a near-tie shows a tiny gap there
            static const bool margin = std::getenv("STRATA_DEC_MARGIN") != nullptr;
            if (margin && tok != forced[(size_t) gen])
                std::fprintf(stderr, "margin: step %d argmax %d (%.4f) forced %d (%.4f) gap %.4f\n", gen, tok, lg[(size_t) tok],
                             forced[(size_t) gen], lg[(size_t) forced[(size_t) gen]], lg[(size_t) tok] - lg[(size_t) forced[(size_t) gen]]);
            tok = forced[(size_t) gen];
        }
        std::printf("%d\n", tok);
        std::fflush(stdout);
        if (std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end() || tok == (int) G.eos) { ++gen; break; }
        if (gen + 1 == a.n_predict) { ++gen; break; }
        if (!step(tok, pos) || !dense.logits(0, 1, lg.data())) { std::fprintf(stderr, "ds41_generate: decode: %s\n", ferr.c_str()); return 1; }
        ++pos;
    }
    const double dec_ms = now_ms() - td0;
    if (prof_decode) Ds41MoeTier::profiler(false);
    const Ds41MoeStats st = tier.stats();
    std::fprintf(stderr, "decode: %d tokens in %.2f s = %.2f tok/s; experts %lld hit / %lld cpu / %lld pcie / %lld file "
                 "(%.1f%% VRAM hits), cpu %.0f ms, file %.0f ms; spec reads %lld issued / %lld used; prefetch %lld issued / "
                 "%lld outside the arena / %lld used\n",
                 gen, dec_ms / 1000.0, 1000.0 * std::max(0, gen - 1) / std::max(dec_ms, 1e-9), (long long) st.hits,
                 (long long) st.cpu, (long long) st.pcie, (long long) st.file_tier,
                 100.0 * st.hits / std::max<int64_t>(1, st.lookups()), st.cpu_ms, st.file_ms,
                 (long long) st.spec_issued, (long long) st.spec_hits, (long long) st.prefetch_issued,
                 (long long) st.prefetch_dummy, (long long) st.prefetched_useful);
    if (!a.peer_devices.empty())
        std::fprintf(stderr, "peer: %zu device(s) hold %lld experts, computed %lld decode rows\n", a.peer_devices.size(),
                     (long long) tier.peer_experts(), (long long) tier.peer_rows());
    if (spec)
        std::fprintf(stderr, "DSpark: %ld verify passes, %.3f accepted drafts/pass = %.3f tokens/pass, %.1f%% full, %ld re-fed; "
                     "verify %.1f ms/pass, draft %.2f ms/pass (depth %d)\n", sp_pass, (double) sp_acc / std::max(1L, sp_pass),
                     1.0 + (double) sp_acc / std::max(1L, sp_pass), 100.0 * sp_full / std::max(1L, sp_pass), sp_refeed,
                     sp_vms / std::max(1L, sp_pass), sp_dms / std::max(1L, sp_pass), spec_depth);
    if (spec)
        std::fprintf(stderr, "pass_v split (all %ld passes, %.2f tokens each): dense %.1f ms, tier %.1f ms, taps %.1f ms, "
                     "logits %.1f ms per pass\n", pv_n, (double) pv_tok / std::max(1L, pv_n), pv_ms[0] / std::max(1L, pv_n),
                     pv_ms[1] / std::max(1L, pv_n), pv_ms[2] / std::max(1L, pv_n), pv_ms[3] / std::max(1L, pv_n));
    if (spec)
        std::fprintf(stderr, "pass_v dense split: begin %.2f, predict %.2f, attn_router %.2f, finish %.2f ms per pass\n",
                     pvd_ms[0] / std::max(1L, pv_n), pvd_ms[1] / std::max(1L, pv_n), pvd_ms[2] / std::max(1L, pv_n), pvd_ms[3] / std::max(1L, pv_n));
    if (g_routes.on()) {   // calibration: the routing counts (v41_rank.py's stats/LNN.json) after the decode
        std::string rerr;
        if (!g_routes.write(rerr)) std::fprintf(stderr, "routes: %s\n", rerr.c_str());
    }
    if (dump_main) std::fclose(dump_main);
    if (file_pred) {
        std::fprintf(stderr, "file-pred: %lld file-tier routes", fp_miss[0]);
        for (int k = 1; k <= kFP; ++k)
            std::fprintf(stderr, " | k=%d: %lld scored, in top4 %.1f%% top8 %.1f%% top16 %.1f%%", k, fp_miss[k],
                         100.0 * qfp_hit[k][0] / std::max(1LL, fp_miss[k]), 100.0 * qfp_hit[k][1] / std::max(1LL, fp_miss[k]),
                         100.0 * qfp_hit[k][2] / std::max(1LL, fp_miss[k]));
        std::fprintf(stderr, "\nfile-pred candidates (file-tier experts in the top 4/8/16, all layers):");
        for (int k = 1; k <= kFP; ++k)
            std::fprintf(stderr, " k=%d %lld/%lld/%lld", k, fp_cand[k][0], fp_cand[k][1], fp_cand[k][2]);
        std::fprintf(stderr, "\n");
    }
    tier.close();
    return 0;
}
