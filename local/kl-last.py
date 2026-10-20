#!/usr/bin/env python3
"""kl-last.py A.f32 B.f32 [...] - KL(A || B) of two last-row logit dumps (--dump-last) and whether their argmax agree."""
import sys, numpy as np
def lsm(p):
    x = np.fromfile(p, np.float32).astype(np.float64); x -= x.max(); return x - np.log(np.exp(x).sum())
a, b = lsm(sys.argv[1]), lsm(sys.argv[2])
pa = np.exp(a)
print(f"KL(exact||bounded) {float((pa * (a - b)).sum()):.5f}  top1 {'same' if a.argmax() == b.argmax() else 'DIFF'}  p_exact(top1) {pa.max():.3f}  p_bounded(exact top1) {float(np.exp(b[a.argmax()])):.3f}")
