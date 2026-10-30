#!/usr/bin/env python3
"""rocprof_summary.py TRACE_DIR [N] - rocprofv3 kernel-trace CSVs (--kernel-trace --stats): the top-N kernels by total time
over the whole run, and the same split into time windows separated by the engine's idle gaps (> 2 s): window 0 = the load,
then one per request (the 98K read, the +1000 append, the decodes)."""
import csv, glob, sys
from collections import defaultdict
d = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 18
rows = []
for f in glob.glob(f'{d}/**/*kernel_trace.csv', recursive=True):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Kernel_Name']))
if not rows:
    sys.exit('no kernel_trace.csv under ' + d)
rows.sort()
def report(sel, title):
    tot = defaultdict(lambda: [0, 0])
    for s, e, k in sel:
        tot[k][0] += e - s; tot[k][1] += 1
    T = sum(v[0] for v in tot.values())
    span = (sel[-1][1] - sel[0][0]) if sel else 0
    print(f'== {title}: {len(sel)} dispatches, GPU busy {T / 1e9:.2f} s of {span / 1e9:.2f} s span')
    for k, (t, n) in sorted(tot.items(), key=lambda x: -x[1][0])[:N]:
        print(f'  {100 * t / T:5.1f}%  {t / 1e6:9.1f} ms  {n:8d} x {t / n / 1e3:8.1f} us  {k[:110]}')
report(rows, 'whole run')
# windows: split where the gap between consecutive dispatches exceeds 2 s
win, cur = [], [rows[0]]
for r in rows[1:]:
    if r[0] - cur[-1][1] > 2e9:
        win.append(cur); cur = []
    cur.append(r)
win.append(cur)
for i, w in enumerate(win):
    report(w, f'window {i} (starts {(w[0][0] - rows[0][0]) / 1e9:.1f} s)')
