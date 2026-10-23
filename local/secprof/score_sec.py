#!/usr/bin/env python3
"""local/secprof/score_sec.py - the security held-out set's reference and score (local/secprof/run-eval-sec.sh).

    score_sec.py teacher DIR TOKENS.npy OUT.npz    production's logits (DIR/seq*.f32) -> top-256 log-probs + its NLL
    score_sec.py score   DIR TOKENS.npy REF.npz    another model's logits against that reference: KL(ref || ours) over
                                                   ref's top-256 (renormalized there), top-1 agreement with ref's argmax,
                                                   our ppl and ref's ppl on the same next tokens

TOKENS.npy holds the sequences the run used, one row per DIR/seqN.f32 file (run-eval-sec.sh reshapes them).
"""
import json
import sys

import numpy as np

V = 129280
K = 256


def rows(path, n_tok):
    lg = np.memmap(path, np.float32, "r")
    lg = lg.reshape(n_tok, lg.size // n_tok)
    for a in range(0, n_tok - 1, 512):   # rows 0..n-2 predict tokens 1..n-1
        b = min(a + 512, n_tok - 1)
        x = np.asarray(lg[a:b, :V], np.float64)
        x -= x.max(1, keepdims=True)
        yield a, x - np.log(np.exp(x).sum(1, keepdims=True))


def main():
    mode, d, tok_path, ref = sys.argv[1:5]
    toks = np.load(tok_path)
    if mode == "teacher":
        idx, lp, nll = [], [], 0.0
        for s, t in enumerate(toks):
            for a, ls in rows("%s/seq%d.f32" % (d, s), len(t)):
                top = np.argpartition(-ls, K, axis=1)[:, :K]
                idx.append(top.astype(np.int32))
                lp.append(np.take_along_axis(ls, top, 1).astype(np.float32))
                nll -= float(ls[np.arange(len(ls)), t[a + 1:a + 1 + len(ls)]].sum())
        idx, lp = np.concatenate(idx), np.concatenate(lp)
        np.savez(ref, idx=idx, lp=lp, nll=nll, tokens=len(idx))
        print(json.dumps({"tokens": len(idx), "ppl": float(np.exp(nll / len(idx)))}))
        return
    z = np.load(ref)
    tidx, tlp = z["idx"], z["lp"]
    nll = kl = 0.0
    top1 = n = 0
    for s, t in enumerate(toks):
        for a, ls in rows("%s/seq%d.f32" % (d, s), len(t)):
            m = len(ls)
            ti, tl = tidx[n:n + m], tlp[n:n + m].astype(np.float64)
            p = np.exp(tl - tl.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            kl += float((p * (np.log(p + 1e-30) - np.take_along_axis(ls, ti, 1))).sum())
            top1 += int((np.argmax(ls, 1) == ti[np.arange(m), np.argmax(tl, 1)]).sum())
            nll -= float(ls[np.arange(m), t[a + 1:a + 1 + m]].sum())
            n += m
    assert n == len(tidx), (n, len(tidx))
    print(json.dumps({"tokens": n, "ppl": float(np.exp(nll / n)), "ref_ppl": float(np.exp(float(z["nll"]) / n)),
                      "top1_agree": top1 / n, "kl_top256": kl / n}))


if __name__ == "__main__":
    main()
