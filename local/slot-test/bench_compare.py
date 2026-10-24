#!/usr/bin/env python3
"""bench_compare.py OUT_DIR [OUT_DIR ...] - the batch-window benchmark's report over several arms (bench.py runs):
  rates   per arm and phase: the aggregate tok/s while every request of the phase decodes, each request's own rate
  streams per arm: every concurrent stream against the SAME conversation's solo stream (first divergence), and the
          solo streams of every arm against the first arm's (the single-sequence path must not change)
  rows    per arm (rows/rows.bin): window / sliced rows against the solo rows at the same positions while the fed tokens
          agree - max |logit difference|, argmax agreement (the engine's multi-row noise; the first arm = the floor)"""
import json
import os
import sys

import numpy as np

arms = sys.argv[1:]
res = {a: json.load(open(f'{a}/results.json')) for a in arms}
names = [os.path.basename(a.rstrip('/')) for a in arms]


def phase(r, p):
    return next((x for x in r if x['phase'] == p), None)


print('aggregate tok/s while every request of the phase decodes (per request: its own rate over its span)')
print(f'{"arm":14s} ' + ' '.join(f'{p:>22s}' for p in ('conc2', 'conc3', 'conc4')) + '   solo B1..B4')
for a, nm in zip(arms, names):
    cells = []
    for p in ('conc2', 'conc3', 'conc4'):
        x = phase(res[a], p)
        cells.append(f'{x.get("steady_tok_s", 0):6.2f} ({x.get("steady_tokens", 0):3d} tok/{x.get("steady_s", 0):4.0f} s)' if x else '-'.rjust(22))
    s = phase(res[a], 'solo')
    solo = ' '.join(f'{s.get(q["name"] + "_tok_s", 0):5.2f}' for q in s['requests']) if s else '-'
    print(f'{nm:14s} ' + ' '.join(f'{c:>22s}' for c in cells) + '   ' + solo)
print('\nengine lines (windows):')
for a, nm in zip(arms, names):
    for p in ('conc2', 'conc3', 'conc4'):
        x = phase(res[a], p)
        if not x:
            continue
        for l in x.get('log', []):
            if 'batch windows' in l:
                print(f'  {nm:14s} {p}: {l[len("serve: "):][:120]}')
                break

print('\nstreams: first divergence of each concurrent stream from the same conversation\'s solo stream (tokens equal before it)')


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


for a, nm in zip(arms, names):
    s = phase(res[a], 'solo')
    solo = {q['name']: q['tokens'] for q in s['requests']}
    row = []
    for p in ('conc2', 'conc3', 'conc4'):
        x = phase(res[a], p)
        if not x:
            continue
        for q in x['requests']:
            d = first_div(solo[q['name']], q['tokens'])
            row.append(f'{p}/{q["name"]}: ' + ('same' if d is None else f'@{d}/{len(q["tokens"])}'))
    print(f'  {nm:14s} ' + '  '.join(row))
if len(arms) > 1:
    ref = {q['name']: q['tokens'] for q in phase(res[arms[0]], 'solo')['requests']}
    for a, nm in zip(arms[1:], names[1:]):
        s = phase(res[a], 'solo')
        row = []
        for q in s['requests']:
            d = first_div(ref[q['name']], q['tokens'])
            row.append(f'{q["name"]}: ' + ('same' if d is None else f'@{d}/{len(q["tokens"])}'))
        print(f'  solo streams of {nm} vs {names[0]}: ' + '  '.join(row))

print('\nrows: concurrent rows vs the solo rows at the same positions (while the fed tokens agree): max |dlogit|, argmax agreement')


def load_rows(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, 'rb') as f:
        while True:
            h = np.frombuffer(f.read(16), np.int32)
            if h.size < 4:
                break
            rid, pos, tok, nv = (int(v) for v in h)
            lg = np.frombuffer(f.read(4 * nv), np.float32)
            if lg.size < nv:
                break
            out[(rid, pos)] = (tok, lg)
    return out


for a, nm in zip(arms, names):
    rows = load_rows(f'{a}/rows/rows.bin')
    if not rows:
        print(f'  {nm:14s} no rows.bin')
        continue
    # admission ids: warm = 0, solo B1..B4 = 1..4, conc2 = 5,6, conc3 = 7,8,9, conc4 = 10..13 (bench.py's order)
    ids = {'solo': {'B1': 1, 'B2': 2, 'B3': 3, 'B4': 4}, 'conc2': {'B1': 5, 'B2': 6}, 'conc3': {'B1': 7, 'B2': 8, 'B3': 9},
           'conc4': {'B1': 10, 'B2': 11, 'B3': 12, 'B4': 13}}
    cells = []
    for p in ('conc2', 'conc3', 'conc4'):
        for b, cid in ids[p].items():
            sid = ids['solo'][b]
            mx, agree, n = 0.0, 0, 0
            for (rid, pos), (tok, lg) in rows.items():
                if rid != cid:
                    continue
                ref = rows.get((sid, pos))
                if ref is None or ref[0] != tok:
                    continue
                d = float(np.max(np.abs(ref[1] - lg)))
                mx = max(mx, d)
                agree += int(np.argmax(ref[1]) == np.argmax(lg))
                n += 1
            if n:
                cells.append(f'{p}/{b}: {n} rows, max {mx:.4f}, argmax {100.0 * agree / n:.1f}%')
    print(f'  {nm:14s} ' + '; '.join(cells))
