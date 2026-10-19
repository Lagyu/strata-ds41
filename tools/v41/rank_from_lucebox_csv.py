#!/usr/bin/env python3
"""rank_from_lucebox_csv.py - a ds41_generate --rank file from Lucebox's routing-stats.csv (one row per layer, the
routed-token count of every expert; '#' comment lines first).  Same order as v41_rank.py: hottest (layer, expert)
first - every expert of a uniform-format file costs the same bytes, so the count alone ranks them.

    python3 tools/v41/rank_from_lucebox_csv.py routing-stats.csv rank.bin
"""
import sys

import numpy as np


def main():
    src, out = sys.argv[1], sys.argv[2]
    rows = [np.array([int(x) for x in ln.split(",")], np.int64)
            for ln in open(src) if ln.strip() and not ln.startswith("#")]
    counts = np.stack(rows)                           # [layers, experts]
    L, E = counts.shape
    order = np.argsort(-counts.reshape(-1), kind="stable")
    pairs = np.stack([order // E, order % E], axis=1).astype(np.int32)
    pairs.tofile(out)
    cum = np.cumsum(np.sort(counts.reshape(-1))[::-1]) / counts.sum()
    print(f"{L} layers x {E} experts -> {out}: {len(pairs)} pairs; the hottest 10% cover "
          f"{cum[len(cum) // 10 - 1]:.1%} of the routes, 25% cover {cum[len(cum) // 4 - 1]:.1%}, 50% cover {cum[len(cum) // 2 - 1]:.1%}")


if __name__ == "__main__":
    main()
