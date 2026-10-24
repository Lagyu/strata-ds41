#!/usr/bin/env python3
"""report_orca.py - the Mix-256G comparison's tables from the score files and bench rows (numbers only)."""
import json
import os
import statistics

R = "/home/yuya/llm-src/strata-ds41"
RES = "/home/yuya/ds41/results.jsonl"


def jl(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def row(name, d, ref_ppl=None):
    if not d:
        return "  %-34s (not available)" % name
    return "  %-34s ppl %7.3f  top-1 %5.1f%%  KL %.4f" % (name, d["ppl"], 100 * d["top1_agree"], d["kl_top256"])


print("== held-out text, 8 x 1024 tokens vs the FP8 original (ppl / top-1 agreement / KL on the teacher's top-256)")
H = R + "/local/eval-lucebox/"
print(row("original-weights MXFP4, no ABL", jl(H + "stock-mxfp4/score.json")))
print(row("  (unsloth UD-Q4_K_XL, stock, 10-21)", jl(H + "unsloth-q4/score.json")))
print(row("production MXFP4+ABL", jl(H + "base-3gpu/score.json")))
print(row("orcarouter Mix-256G", jl(H + "orca-mix256/score.json")))
print(row("orcarouter Mix-256G + skip-file 0.10", jl(H + "orca-mix256-skip10/score.json")))
print(row("production MXFP4+ABL + skip-file 0.10", jl(H + "skip10/score.json")))
print(row("Maya-S + our ABL dense (10-24)", jl(H + "mayas-abl/score.json")))
print(row("GSQ-RCO IQ2_XS + our ABL dense", jl(H + "gsq-iq2xs-abl/score.json")))
print("\n== security code set, 16 x 2048 tokens (KL(reference || model), top-1 agreement with the reference, ppl)")
S = R + "/local/secprof/eval/"
for ref, title in (("teacher-stock", "reference = original-weights MXFP4 (no ABL)"), ("teacher-prod", "reference = production MXFP4+ABL")):
    print(" " + title)
    for tag, name in (("prod-sec", "production MXFP4+ABL (test binary)"), ("orca-sec", "orcarouter Mix-256G"), ("mayas-abl", "Maya-S + our ABL dense"),
                      ("gsq-iq2xs-abl", "GSQ-RCO IQ2_XS + our ABL dense")):
        d = jl(S + tag + "/score-%s.json" % ref) or (jl(S + tag + "/score.json") if (tag in ("mayas-abl", "gsq-iq2xs-abl") and ref == "teacher-prod") else None)
        if d:
            print("  %-34s ppl %.4f (ref %.4f, %+.1f%%)  top-1 %5.1f%%  KL %.4f" % (name, d["ppl"], d["ref_ppl"], 100 * (d["ppl"] / d["ref_ppl"] - 1),
                                                                                   100 * d["top1_agree"], d["kl_top256"]))
        else:
            print("  %-34s (not available)" % name)
sk = jl(S + "skip10/score.json")
if sk:
    print("  (other session, 10-24) production + skip-file 0.10 vs production: ppl %.4f, top-1 %.1f%%, KL %.4f" % (sk["ppl"], 100 * sk["top1_agree"], sk["kl_top256"]))
st = jl(S + "stock-ref/score.json")
if st:
    print("  original-weights MXFP4 own ppl on this set: %.4f" % st["ppl"])

print("\n== speed, decode tok/s per prompt (bench.py, thinking off); C = first pass right after the service start, S = measured after a warm-up")
rows = {}
if os.path.exists(RES):
    for line in open(RES):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("label", "").startswith(("C-orca", "S-orca")):
            rows[(r["label"], r["rep"], r["prompt"])] = r
labels = sorted({(k[0], k[1]) for k in rows}, key=lambda x: (x[0][2:], x[0][0] != "C", x[1]))
print("  %-16s %4s | %7s %7s %7s %7s | %8s | %s" % ("label", "rep", "code", "prose", "ja", "after4K", "4K pref", "accept"))
for lab, rep in labels:
    g = lambda p, k: (rows.get((lab, rep, p)) or {}).get(k)
    dec = [g(p, "decode_tok_s") for p in ("code", "prose", "ja", "4k")]
    acc = [g(p, "accept_rate") for p in ("code", "prose", "ja", "4k")]
    print("  %-16s %4d | %7s %7s %7s %7s | %8s | %s" % (lab, rep, *(("%.1f" % x) if x else "-" for x in dec),
                                                    ("%.1f" % g("4k", "prefill_tok_s")) if g("4k", "prefill_tok_s") else "-",
                                                    "/".join(("%.2f" % x) if x is not None else "-" for x in acc)))
for cfg in sorted({l[2:] for l, _ in labels if l.startswith("S-")}):
    for p in ("code", "prose", "ja", "4k"):
        v = [rows[(l, r, p)]["decode_tok_s"] for (l, r) in labels if l == "S-" + cfg and r >= 1 and (l, r, p) in rows and rows[(l, r, p)].get("decode_tok_s")]
        if v:
            print("  S-%s %-5s decode over the repeated reps (rep >= 1): %s -> median %.1f" % (cfg, p, ", ".join("%.1f" % x for x in v), statistics.median(v)))

print("\n== refusal check (4 prompts, thinking off): answered / refused")
for name, f in (("original-weights MXFP4 (no ABL)", "orca-refusal-stock.txt"), ("production MXFP4+ABL", "orca-refusal-prod.txt"),
                ("orcarouter Mix-256G", "orca-refusal-mix.txt")):
    try:
        t = open(R + "/logs/" + f).read().splitlines()
        print("  %-34s %d answered, %d refused" % (name, sum(1 for x in t if x.startswith("answered")), sum(1 for x in t if x.startswith("REFUSED"))))
    except OSError:
        print("  %-34s (not available)" % name)
