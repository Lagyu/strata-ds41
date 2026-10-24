#!/usr/bin/env python3
"""Speed table from results.jsonl: one row per label, prefill / decode tok/s per prompt (last run of each).

  table.py LABEL [LABEL ...]        (e.g. table.py final-rerun gsq-default gsq-exact)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = ["code", "prose", "ja", "4k"]


def main():
    labels = sys.argv[1:]
    rows = [json.loads(l) for l in open(HERE / "results.jsonl") if l.strip()]
    last = {}
    for r in rows:
        if r.get("label") in labels:
            last[(r["label"], r.get("prompt"))] = r
    print(f"{'label':18s}" + "".join(f"{p + ' prefill':>14s}{p + ' decode':>13s}" for p in PROMPTS) + "   accept(code/prose/ja/4k)")
    for lab in labels:
        line = f"{lab:18s}"
        acc = []
        for p in PROMPTS:
            r = last.get((lab, p))
            if not r:
                line += f"{'-':>14s}{'-':>13s}"
                acc.append("-")
                continue
            pf = r.get("prefill_tok_s")
            dc = r.get("decode_tok_s")
            line += f"{(f'{pf:.1f}' if pf else '-'):>14s}{(f'{dc:.1f}' if dc else '-'):>13s}"
            a = r.get("accept_rate")
            acc.append(f"{a:.2f}" if isinstance(a, (int, float)) else "-")
        print(line + "   " + "/".join(acc))


if __name__ == "__main__":
    main()
