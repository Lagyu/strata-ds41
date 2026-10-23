#!/usr/bin/env python3
"""rank_from_counts.py - a ds41_generate --rank file from the server's own route counts (STRATA_ROUTE_COUNTS: counts.csv
and the counts-*.csv the launcher rotates at each start; Lucebox CSV format, one row per layer).  The counts are summed;
experts the traffic never routed keep their order from a fallback rank (--fallback, default the Lucebox rank), so the
tail is never random.  Prints what share of the counted routes each residency tier would serve under the new rank and
under the fallback (the production tiers: 762 VRAM slots, 5425 arena experts, 2400 on the peers, the rest on the file).

    python3 tools/v41/rank_from_counts.py /home/yuya/datasets/prod-route-counts rank-traffic.bin
    (then: "--rank", "<path>" in the config's args, and a service restart)

Needs a few days of traffic to be worth it: on 2026-10-22 only the warmups' counts survived the day's restarts."""
import glob
import os
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
fallback = sys.argv[3] if len(sys.argv) > 3 else '/home/yuya/models/dsv41-1017-strata/rank-lucebox.bin'
files = sorted(glob.glob(os.path.join(src, 'counts*.csv'))) if os.path.isdir(src) else [src]
tot = None
for f in files:
    rows = [np.array([int(x) for x in ln.split(',')], np.int64) for ln in open(f) if ln.strip() and not ln.startswith('#')]
    c = np.stack(rows)
    print(f'{os.path.basename(f)}: {c.shape[0]} layers x {c.shape[1]} experts, {c.sum():,} routed events (~{c.sum() / (6 * c.shape[0]):,.0f} tokens)')
    tot = c if tot is None else tot + c
L, E = tot.shape
flat = tot.reshape(-1).astype(np.float64)
fb = np.fromfile(fallback, np.int32).reshape(-1, 2)
fb_idx = fb[:, 0].astype(np.int64) * E + fb[:, 1]
fb_pos = np.empty(L * E, np.int64)
fb_pos[fb_idx] = np.arange(L * E)
# counted experts first (most routed first), then the uncounted in the fallback's order
order = np.lexsort((fb_pos, -flat))
pairs = np.stack([order // E, order % E], axis=1).astype(np.int32)
pairs.tofile(out)
tiers = [('VRAM head', 0, 762), ('arena', 762, 762 + 5425), ('peers', 762 + 5425, 762 + 5425 + 2400), ('file', 762 + 5425 + 2400, L * E)]
print(f'\n{out}: {len(pairs)} pairs ({int((flat > 0).sum())} experts were routed at least once)')
print(f"{'tier':10s} {'new rank':>9s} {'fallback':>9s}   (share of the counted routes)")
for name, a, b in tiers:
    print(f'{name:10s} {flat[order[a:b]].sum() / max(1, flat.sum()):9.1%} {flat[fb_idx[a:b]].sum() / max(1, flat.sum()):9.1%}')
