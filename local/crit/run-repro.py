#!/usr/bin/env python3
"""run-repro.py OUT_DIR PAYLOAD_DIR LABEL [ENV=VALUE ...] [-- args] - warm (U1, 4 tokens) then a 16K read (L[:16384], 8 tokens)."""
import json, os, re, subprocess, sys, time
import numpy as np
argv = sys.argv[1:]; extra_args = []
if '--' in argv: extra_args = argv[argv.index('--') + 1:]; argv = argv[:argv.index('--')]
out_dir, payload, label = argv[0], argv[1], argv[2]
extra_env = dict(kv.split('=', 1) for kv in argv[3:])
R = '/home/yuya/llm-src/strata-ds41'
cfg = json.load(open(os.environ.get('PROBE_CONFIG') or f'{R}/local/ds41-yuya-3gpu.json'))
os.makedirs(out_dir, exist_ok=True)
pc = {k: np.fromfile(f'{payload}/{k}.i32', np.int32).tolist() for k in ('U1', 'L')}
env = dict(os.environ); env['LD_LIBRARY_PATH'] = '/opt/rocm/lib'
env.update({k: str(v) for k, v in cfg['env'].items() if k != 'STRATA_ROUTE_COUNTS'}); env.update(extra_env)
cmd = [f'{R}/local/ds41-engine.sh', '--serve'] + list(cfg['args']) + extra_args
log = open(f'{out_dir}/engine.log', 'w')
proc = subprocess.Popen(cmd, cwd=R, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
while True:
    ln = proc.stdout.readline()
    if not ln: sys.exit('engine died before READY')
    if ln.startswith('READY'): break
def gen(name, ids, n):
    t = time.time(); proc.stdin.write(f'GEN {n} temperature=0 ' + ','.join(map(str, ids)) + '\n'); proc.stdin.flush()
    toks = []
    while True:
        ln = proc.stdout.readline()
        if not ln: print(f'[{label}] {name}: engine died mid-request', flush=True); sys.exit(1)
        if ln.startswith('T '): toks.append(int(ln[2:]))
        elif ln.startswith('DONE') or ln.startswith('ERR'): break
    print(f'[{label}] {name:10s} {len(ids)} tokens -> {len(toks)} out in {time.time() - t:.1f} s', flush=True)
gen('warm', pc['U1'], 4)
gen('read16k', pc['L'][:16384], 8)
gen('append', pc['L'][:16384 + 300], 8)
proc.stdin.write('QUIT\n'); proc.stdin.flush()
try: proc.wait(timeout=120)
except subprocess.TimeoutExpired: proc.kill()
print(f'[{label}] done', flush=True)
