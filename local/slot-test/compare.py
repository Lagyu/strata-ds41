#!/usr/bin/env python3
"""compare.py PREFIX_DIR ENDPARK_DIR SLOTS_DIR - the checks of the three runs (any dir may be '-' = skip)."""
import json
import sys

import numpy as np

bad = 0


def load(d):
    return {r['name']: r for r in json.load(open(f'{d}/results.json'))}


def logits(d, r):
    return np.fromfile(f"{d}/dump/req{r['dump_index']:03d}.f32", np.float32)


def kl(ref, x):
    px = np.exp(ref - ref.max()); px /= px.sum(); py = np.exp(x - x.max()); py /= py.sum()
    return float(np.sum(px * (np.log(px + 1e-30) - np.log(py + 1e-30))))


def check(cond, what):
    global bad
    bad += not cond
    print(('ok   ' if cond else 'FAIL ') + what)


def cmp_logits(d, R, a, b):
    x, y = logits(d, R[a]), logits(d, R[b])
    same = bool((x == y).all())
    print(f'     {a} vs {b}: max|d| {float(np.abs(x - y).max()):.4g}, KL {kl(y, x):.3g}, bit-identical {same}')
    return same


pfx, endp, slots = sys.argv[1:4]
if pfx != '-':
    R = load(pfx)
    print('== prefix')
    check(any('shared prefix of 6000 tokens parked' in l for l in R['B_split']['log']), 'B_split: the read was split and P parked as a shared prefix')
    check(R['B_split'].get('resume', 0) == 0, 'B_split read in full')
    check(R['C_prefix'].get('resume', 0) == 6000 and any('shared prefix' in l and 'unparked' in l for l in R['C_prefix']['log']),
          'C_prefix: RESUME 6000 from the shared prefix')
    check(cmp_logits(pfx, R, 'C_prefix', 'B_split'), 'C_prefix logits == B_split (the same tail from the same prefix rows)')
    check(R['C_prefix']['tokens'] == R['B_split']['tokens'], 'C_prefix first token == B_split first token')
    print(f"     (D_direct vs C_prefix: two independent reads of P, informational)")
    cmp_logits(pfx, R, 'C_prefix', 'D_direct') if R['C_prefix']['n_prompt'] == R['D_direct']['n_prompt'] else None
    check(R['E_prefix'].get('resume', 0) == 6000, 'E_prefix: RESUME 6000 (the prefix served a second request)')
    check(R['P_regen'].get('resume', 0) == 6000 and any('shared prefix' in l for l in R['P_regen']['log']) and R['P_regen']['wall'] < 3,
          f"P_regen: the prefix itself regenerates from the park without a read ({R['P_regen']['wall']} s)")
    check(not any('dropped the 6000-token shared prefix' in l for r in R.values() for l in r['log']), 'the prefix was never evicted (PARK_SLOTS=2)')
    for n in ('B_split', 'C_prefix', 'E_prefix'):
        print(f"     {n}: {R[n]['wall']} s")
if endp != '-':
    R = load(endp)
    print('== endpark')
    W, ans = R['W60']['n_prompt'], R['W60']['tokens']
    check(R['W40'].get('resume', 0) == W and any('prompt end' in l for l in R['W40']['log']), 'W40: the same prompt again from the park, at its prompt end')
    check(R['W40']['tokens'] == ans[:40], 'W40 (regen from the park) == the uninterrupted run\'s first 40 tokens')
    c = R['W_cont']
    check(W + 38 <= c.get('resume', 0) <= W + 40 and any('unparked' in l and 'prompt end' not in l and 'tail' not in l for l in c['log']),
          f"W_cont resumed at the parked conversation's END (RESUME {c.get('resume')}, end {W + 40})")
    n = next((i for i, (x, y) in enumerate(zip(c['tokens'], ans[41:])) if x != y), len(c['tokens']))
    check(n >= 5, f"W_cont continues like the uninterrupted run (first difference at token {n} of {len(c['tokens'])}; later near-tie flips are expected)")
    t = R['W_tail']
    check(any('tail: re-feeds' in l and 'below its end' in l for l in t['log']) and t['expect'][1] - 17 <= t.get('resume', 0) <= t['expect'][1],
          f"W_tail: resumed within rollback_max of the parked END (RESUME {t.get('resume')}, end ~{t['expect'][1]})")
    check(len(t['tokens']) == 4, 'W_tail decoded its 4 tokens')
if len(sys.argv) > 4 and sys.argv[4] != '-':
    R = load(sys.argv[4])
    print('== prefix2 (a new engine start with the saved prefix files)')
    c = R['C_loaded']
    check(c.get('resume', 0) == 6000 and any('shared prefix' in l and 'unparked' in l for l in c['log']), f"C_loaded: RESUME {c.get('resume')} from the prefix loaded from file")
    check(c['wall'] < 25, f"C_loaded read only its tail ({c['wall']} s)")
    check(R['P_regen2'].get('resume', 0) == 6000 and R['P_regen2']['wall'] < 3, 'P_regen2: the loaded prefix regenerates from its saved logits')
if len(sys.argv) > 5 and sys.argv[5] != '-':
    R = load(sys.argv[5])
    print('== cancel (STOP during a long read)')
    c = R['long_cut']
    cut = next((l for l in c['log'] if 'prompt read cancelled after' in l), '')
    n_kept = int(cut.split('after ')[1].split(' of')[0]) if cut else -1
    check('cancel' in c.get('end', '') and c['wall'] < 25 and 0 < n_kept < c['n_prompt'], f"long_cut: cancelled in {c['wall']} s with {n_kept} of {c['n_prompt']} tokens kept ({c.get('end', '')[:30]})")
    r = R['long_resend']
    check(r.get('resume', 0) == n_kept and len(r['tokens']) == 4, f"long_resend: continued from the kept part (RESUME {r.get('resume')}, kept {n_kept})")
    a = R['long_again']
    check(a.get('resume', 0) >= r['n_prompt'] - 2 and len(a['tokens']) == 4, f"long_again: from the park at its END (RESUME {a.get('resume')}, end ~{r['n_prompt']})")
if slots != '-':
    R = load(slots)
    print('== slots')
    for a, b in (('S1_par', 'S1_seq'), ('S2_par', 'S2_seq'), ('S1_stop', 'S1_seq')):
        check(R[a]['tokens'] == R[b]['tokens'], f'{a} tokens == {b} tokens ({len(R[a]["tokens"])} vs {len(R[b]["tokens"])})')
    for a, b in (('S1_seq', 'S1_solo'), ('S2_seq', 'S2_solo')):
        same = R[a]['tokens'] == R[b]['tokens']
        n = next((i for i, (x, y) in enumerate(zip(R[a]['tokens'], R[b]['tokens'])) if x != y), len(R[a]['tokens']))
        print(f'     {a} vs {b} (a regen from the park vs the solo run): identical {same}, first difference at token {n}')
    st = R['S2_stop']
    check('cancel' in st.get('end', '') and 10 <= len(st['tokens']) <= 16, f"S2_stop: BSTOP ended it with cancel after {len(st['tokens'])} tokens ({st.get('end', '')})")
    check(R['S1_stop']['tokens'] == R['S1_seq']['tokens'], 'S1_stop: the other slot ran on to its end unchanged')
    cut, cont, solo = R['S3_cut'], R['S3_cont'], R['S3_solo']
    joined = cut['tokens'] + cont['tokens']
    check(cont['expect'] - 1 <= cont.get('resume', 0) <= cont['expect'], f"S3_cont resumed the stopped request's conversation (RESUME {cont.get('resume')}, expected ~{cont['expect']})")
    n = next((i for i, (x, y) in enumerate(zip(joined, solo['tokens'])) if x != y), len(joined))
    print(f"     S3 stopped + continued in a slot vs the plain run: identical {joined == solo['tokens'][:len(joined)]}, first difference at token {n} of {len(joined)} (informational: a 1-row pass rounds like any continuation)")
    for r in R.values():
        if r['kind'].startswith('BGEN') and r.get('wall'):
            print(f"     {r['name']}: {len(r['tokens'])} tokens in {r['wall']} s")
print('ALL OK' if bad == 0 else f'{bad} CHECKS FAILED')
sys.exit(1 if bad else 0)
