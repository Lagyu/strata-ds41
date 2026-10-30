#!/usr/bin/env python3
"""cmp_lidx.py VEC.bin WMMA.bin - the indexer test's dumped scores (LIDX_DUMP) of the vector kernel vs the matrix-core kernel:
max / mean abs and relative differences over finite scores, and for the three --large checks (nk 32768 / 131072 / 262144, nq 128)
the per-query top-512 set agreement."""
import sys
import numpy as np
a = np.fromfile(sys.argv[1], np.float32); b = np.fromfile(sys.argv[2], np.float32)
assert a.size == b.size, (a.size, b.size)
fin = np.isfinite(a) & np.isfinite(b)
assert np.array_equal(np.isfinite(a), np.isfinite(b)), "the -inf pattern differs"
d = np.abs(a[fin] - b[fin]); s = np.abs(a[fin]) + 1e-6
print(f"scores {a.size} (finite {fin.sum()}): max abs diff {d.max():.3e}, mean {d.mean():.3e}, max rel {np.max(d / s):.3e}, "
      f"> 1e-3 rel: {(d / s > 1e-3).sum()}, bit-identical {(a[fin] == b[fin]).mean() * 100:.2f}%")
# the large checks are the last ones: nq = 128 each, nk = 32768, 131072, 262144 (row-major [nq][nk])
off = a.size
for nk in (262144, 131072, 32768):
    n = 128 * nk; off -= n
    A = a[off:off + n].reshape(128, nk); B = b[off:off + n].reshape(128, nk)
    agree = []
    for q in range(128):
        ta = set(np.argpartition(-A[q], 512)[:512].tolist()); tb = set(np.argpartition(-B[q], 512)[:512].tolist())
        agree.append(len(ta & tb) / 512)
    print(f"nk={nk}: top-512 set agreement mean {np.mean(agree) * 100:.3f}% min {np.min(agree) * 100:.2f}%")
