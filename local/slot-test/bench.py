#!/usr/bin/env python3
"""bench.py OUT_DIR PAYLOAD_DIR LABEL [K] [ENV=VALUE ...] [-- extra engine args]

The batch-window benchmark of one `ds41_generate --serve` (line protocol, greedy, K tokens a request; default 120):
  warm   WARM as a solo GEN (20 tokens): the VRAM expert cache opens
  solo   B1..B4 one after another as solo GENs (the references: their token streams and rates)
  conc2  B1 + B2 as BGEN 0/1 at once      } every conversation resumes from its park at the prompt's end, so the
  conc3  B1 + B2 + B3 as BGEN 0/1/2       } context of every decoded token is the one its solo run had: the streams
  conc4  B1 .. B4 as BGEN 0..3            } are comparable token for token
Per phase: every request's tokens with their arrival times, the aggregate rate while ALL of the phase's requests
decode (from the last admission to the first BDONE) and over the whole phase, and the engine's own figures (serve: N
tokens in ..., serve: batch windows ...).  STRATA_SERVE_ROWDUMP=OUT_DIR/rows makes the engine write every decode row's
logits (bench_compare.py reads them: window rows against the solo rows at the same positions).
Engine command, arguments and environment: the production config's (local/ds41-yuya-3gpu.json of BENCH_TREE or the
production tree) plus the extras; DS41_EXE= picks the binary."""
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

argv = sys.argv[1:]
extra_args = []
if '--' in argv:
    extra_args = argv[argv.index('--') + 1:]
    argv = argv[:argv.index('--')]
out_dir, payload, label = argv[0], argv[1], argv[2]
rest = argv[3:]
K = 120
if rest and rest[0].isdigit():
    K = int(rest[0])
    rest = rest[1:]
extra_env = dict(kv.split('=', 1) for kv in rest)
R = os.environ.get('BENCH_TREE') or '/home/yuya/llm-src/strata-ds41'
cfg = json.load(open(f'{R}/local/ds41-yuya-3gpu.json'))
os.makedirs(f'{out_dir}/rows', exist_ok=True)
pc = {k: np.fromfile(f'{payload}/{k}.i32', np.int32).tolist() for k in json.load(open(f'{payload}/sizes.json'))}

env = dict(os.environ)
env['LD_LIBRARY_PATH'] = '/opt/rocm/lib' + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
env.update({k: str(v) for k, v in cfg['env'].items() if k != 'STRATA_ROUTE_COUNTS'})   # never the production counts
env.update(STRATA_TEXT_LOG='0', STRATA_SERVE_ROWDUMP=f'{out_dir}/rows')
env.update(extra_env)
args = list(cfg['args']) + extra_args
cmd = [f'{R}/local/ds41-engine.sh', '--serve'] + args
log = open(f'{out_dir}/engine.log', 'w')
proc = subprocess.Popen(cmd, cwd=R, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
t_start = time.time()
info_lines = []
while True:
    ln = proc.stdout.readline()
    if not ln:
        sys.exit('engine died before READY (see engine.log)')
    if ln.startswith('INFO'):
        info_lines.append(ln.strip())
    if ln.startswith('READY'):
        break
print(f'[{label}] ready after {time.time() - t_start:.0f} s  {" | ".join(info_lines)}', flush=True)

log_pos = 0
LOG_PREFIX = ('serve: prompt ', 'serve: reuse', 'serve: unparked', 'serve: parked', 'serve: tail resume', 'serve: batch windows',
              'serve: DSpark', 'serve: lookup', 'serve: VRAM', 'serve: resumed', 'serve: batch window failed', 'serve: ')


def log_lines():
    """the engine's numeric / decision lines written since the last call (never a prompt's text: the engine logs none)"""
    global log_pos
    log.flush()
    with open(f'{out_dir}/engine.log', 'rb') as f:
        f.seek(log_pos)
        text = f.read().decode('utf-8', 'replace')
        log_pos = f.tell()
    keep = []
    for l in text.splitlines():
        if re.match(r'serve: \d+ tokens in|serve: prompt \d+ tokens|serve: batch windows|serve: DSpark \d+ passes|serve: lookup|'
                    r'serve: unparked|serve: parked|serve: reuse|serve: tail resume|serve: batch window failed|serve: resumed|'
                    r'serve: \d+ conversation slots|serve: VRAM', l):
            keep.append(l[:200])
    return keep


def send(s):
    proc.stdin.write(s + '\n')
    proc.stdin.flush()


def readline():
    ln = proc.stdout.readline()
    if not ln:
        sys.exit('engine died mid-request')
    return ln.strip()


results = []


class Req:
    def __init__(self, name, slot, ids, n):
        self.name, self.slot, self.ids, self.n = name, slot, ids, n
        self.toks, self.times, self.info = [], [], {}
        self.done = False
        self.t0 = time.time()
        self.t_adm = None     # admitted (BADM / the solo's first token)
        self.t_end = None


def rates(reqs, t_all, t_first_end):
    """tokens per second: aggregate while every request decodes [t_all, t_first_end], and per request over its own span"""
    out = {}
    if t_all is not None and t_first_end is not None and t_first_end > t_all:
        n = sum(sum(1 for t in r.times if t_all < t <= t_first_end) for r in reqs)
        out['steady_tok_s'] = round(n / (t_first_end - t_all), 2)
        out['steady_s'] = round(t_first_end - t_all, 1)
        out['steady_tokens'] = n
    for r in reqs:
        if len(r.times) >= 2:
            out[f'{r.name}_tok_s'] = round((len(r.times) - 1) / (r.times[-1] - r.times[0]), 2)
    return out


def phase_solo(name, prompts, n):
    reqs = []
    for pname in prompts:
        r = Req(pname, -1, pc[pname], n)
        send(f'GEN {n} temperature=0 ' + ','.join(map(str, r.ids)))
        while True:
            ln = readline()
            now = time.time()
            if ln.startswith('T '):
                r.toks.append(int(ln[2:])); r.times.append(now)
                if r.t_adm is None:
                    r.t_adm = now
            elif ln.startswith('RESUME'):
                r.info['resume'] = int(ln.split()[1])
            elif ln.startswith('DONE') or ln.startswith('ERR'):
                r.info['end'] = ln[:80]
                r.t_end = now
                break
        r.info['log'] = log_lines()
        r.info['wall'] = round(r.t_end - r.t0, 1)
        reqs.append(r)
        print(f'[{label}] {name:6s} {pname:4s} GEN   prompt {len(r.ids)} resume {r.info.get("resume", 0)} first token after '
              f'{(r.t_adm or r.t_end) - r.t0:5.1f} s, {len(r.toks)} tokens, '
              f'{(len(r.times) - 1) / max(1e-9, r.times[-1] - r.times[0]) if len(r.times) > 1 else 0:5.2f} tok/s  {r.info["end"][:30]}', flush=True)
        for l in r.info['log']:
            if 'tokens in' in l or 'batch windows' in l or 'DSpark' in l or 'prompt ' in l:
                print('        ' + l[:160], flush=True)
    rec = dict(phase=name, kind='solo', requests=[dict(name=r.name, slot=r.slot, n_prompt=len(r.ids), tokens=r.toks,
               times=[round(t - t_start, 3) for t in r.times], **r.info) for r in reqs])
    rec.update(rates(reqs, None, None))
    results.append(rec)
    json.dump(results, open(f'{out_dir}/results.json', 'w'), indent=1)


def phase_conc(name, prompts, n):
    reqs = [Req(p, i, pc[p], n) for i, p in enumerate(prompts)]

    def take(ln, now):
        f = ln.split()
        if f[0] not in ('BT', 'BDONE'):
            return
        r = next((x for x in reqs if x.slot == int(f[1]) and not x.done), None)
        if r is None:
            print('        stray line ' + ln[:80], flush=True)
            return
        if f[0] == 'BT':
            r.toks.append(int(f[2])); r.times.append(now)
        else:
            r.info['end'] = ln[:80]
            r.done = True
            r.t_end = now

    for r in reqs:
        r.t0 = time.time()
        send(f'BGEN {r.slot} {n} temperature=0 ' + ','.join(map(str, r.ids)))
        while True:
            ln = readline()
            now = time.time()
            if ln.startswith('BT ') or ln.startswith('BDONE '):
                take(ln, now)
            elif ln.startswith('T '):
                r.toks.append(int(ln[2:])); r.times.append(now)
            elif ln.startswith('RESUME'):
                r.info['resume'] = int(ln.split()[1])
            elif ln.startswith('DONE'):
                r.info['adm'] = ln[:80]
            elif ln.startswith('BADM'):
                r.t_adm = now
                if ln.split()[2] == '0':
                    r.done = True
                    r.t_end = now
                    r.info['end'] = 'ended at admission'
                break
            elif ln.startswith('ERR'):
                r.info['end'] = ln[:80]
                r.done = True
                r.t_end = now
                break
        print(f'[{label}] {name:6s} {r.name:4s} BGEN{r.slot} prompt {len(r.ids)} resume {r.info.get("resume", 0)} admitted in {now - r.t0:5.1f} s', flush=True)
    t_all = max((r.t_adm or r.t0) for r in reqs)   # every request of the phase is decoding from here
    while any(not r.done for r in reqs):
        take(readline(), time.time())
    t_first_end = min(r.t_end for r in reqs)
    lines = log_lines()
    rec = dict(phase=name, kind='conc', t_all=round(t_all - t_start, 3), t_first_end=round(t_first_end - t_start, 3), log=lines,
               requests=[dict(name=r.name, slot=r.slot, n_prompt=len(r.ids), tokens=r.toks, times=[round(t - t_start, 3) for t in r.times],
                              wall=round(r.t_end - r.t0, 1), **r.info) for r in reqs])
    rec.update(rates(reqs, t_all, t_first_end))
    results.append(rec)
    json.dump(results, open(f'{out_dir}/results.json', 'w'), indent=1)
    print(f'[{label}] {name:6s} steady state {rec.get("steady_s", 0):5.1f} s, {rec.get("steady_tokens", 0)} tokens = '
          f'{rec.get("steady_tok_s", 0):5.2f} tok/s aggregate; per request ' +
          ', '.join(f'{r.name} {rec.get(r.name + "_tok_s", 0):.2f}' for r in reqs), flush=True)
    for l in lines:
        if 'tokens in' in l or 'batch windows' in l or 'window failed' in l:
            print('        ' + l[:160], flush=True)


phase_solo('warm', ['WARM'], 20)
phase_solo('solo', ['B1', 'B2', 'B3', 'B4'], K)
phase_conc('conc2', ['B1', 'B2'], K)
phase_conc('conc3', ['B1', 'B2', 'B3'], K)
phase_conc('conc4', ['B1', 'B2', 'B3', 'B4'], K)
send('QUIT')
try:
    proc.wait(timeout=120)
except subprocess.TimeoutExpired:
    proc.kill()
print(f'[{label}] done', flush=True)
