#!/usr/bin/env python3
"""dump-cmp.py DIR_A DIR_B - compare the DS41_DUMP_LAYER tensors of two runs: identical / first differing element."""
import sys, os, numpy as np
a, b = sys.argv[1], sys.argv[2]
for name in ["o_fn", "o_ids", "o_wts", "hap", "post_f", "comb_f", "pre_state", "shexp", "x_state", "raw"]:
    fa, fb = os.path.join(a, name + ".bin"), os.path.join(b, name + ".bin")
    if not (os.path.exists(fa) and os.path.exists(fb)): continue
    dt = np.int32 if name == "o_ids" else (np.uint16 if name == "raw" else np.float32)
    x, y = np.fromfile(fa, dtype=dt), np.fromfile(fb, dtype=dt)
    if x.shape != y.shape: print(f"{name:10s} shape {x.shape} vs {y.shape}"); continue
    d = np.nonzero(x != y)[0]
    if len(d) == 0: print(f"{name:10s} identical ({x.size})")
    else:
        i = d[0]
        print(f"{name:10s} DIFFER: {len(d)}/{x.size} elements; first at {i}: {x[i]!r} vs {y[i]!r}" + ("" if dt != np.float32 else f" (max |d| {np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))):.3e})"))
