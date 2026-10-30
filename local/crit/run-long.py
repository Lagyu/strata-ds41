#!/usr/bin/env python3
"""run-long.py OUT_DIR PAYLOAD_DIR LABEL [ENV=VALUE ...] [-- extra engine args] - the perf probe at LONG context: one ~98K-token
fresh read (payload piece L), then the agent-like appends (+150 / +400 / +1000 / +60 / +2500, 40 tokens decoded each) on top of
it, then a 200-token decode bench at that context.  Same figures as local/perf-probe/run.py (engine numeric log lines)."""
import json, os, re, subprocess, sys, time
import numpy as np

argv = sys.argv[1:]
extra_args = []
if '--' in argv:
    extra_args = argv[argv.index('--') + 1:]
    argv = argv[:argv.index('--')]
out_dir, payload, label = argv[0], argv[1], argv[2]
extra_env = dict(kv.split('=', 1) for kv in argv[3:])
R = '/home/yuya/llm-src/strata-ds41'
cfg = json.load(open(os.environ.get('PROBE_CONFIG') or f'{R}/local/ds41-yuya-3gpu.json'))
os.makedirs(out_dir, exist_ok=True)
pc = {k: np.fromfile(f'{payload}/{k}.i32', np.int32).tolist() for k in json.load(open(f'{payload}/sizes.json'))}
env = dict(os.environ)
env['LD_LIBRARY_PATH'] = '/opt/rocm/lib' + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
env.update({k: str(v) for k, v in cfg['env'].items() if k != 'STRATA_ROUTE_COUNTS'})
env.update(extra_env)
args = list(cfg['args']) + extra_args
cmd = [f'{R}/local/ds41-engine.sh', '--serve'] + args
log = open(f'{out_dir}/engine.log', 'w')
proc = subprocess.Popen(cmd, cwd=R, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
t0 = time.time()
while True:
    ln = proc.stdout.readline()
    if not ln:
        sys.exit('engine died before READY (see engine.log)')
    if ln.startswith('READY'):
        break
print(f'[{label}] ready after {time.time() - t0:.0f} s', flush=True)
PH = re.compile(r'serve: prompt phases: (\d+) new = (\d+) (\w+) \(([\d.]+) s: dense ([\d.]+), experts ([\d.]+), logits ([\d.]+)\) \+ DSpark tail (\d+)[^(]*\(cache open ([\d.]+) s, seed ([\d.]+) s\)')
PP = re.compile(r'serve: prompt (\d+) tokens \((\d+) reused, ([a-z ]+)\) in ([\d.]+) s')
DC = re.compile(r'serve: (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s \((\w+)[^)]*\); experts: (\d+) hit, (\d+) cpu, (\d+) pcie, (\d+) file')
DS = re.compile(r'serve: DSpark (\d+) passes, ([\d.]+) tokens/pass')
LK = re.compile(r'serve: lookup (\d+)/(\d+) drafts accepted in (\d+) passes')
log_pos = 0
results = []
def figures():
    global log_pos
    log.flush()
    with open(f'{out_dir}/engine.log', 'rb') as f:
        f.seek(log_pos); text = f.read().decode('utf-8', 'replace'); log_pos = f.tell()
    d = {}
    for l in text.splitlines():
        m = PH.search(l)
        if m: d.update(new=int(m[1]), path=m[3], feed=float(m[4]), dense=float(m[5]), experts=float(m[6]), copen=float(m[9]), seed=float(m[10]))
        m = PP.search(l)
        if m: d.update(total=int(m[1]), reused=int(m[2]), prompt_s=float(m[4]))
        m = DC.search(l)
        if m: d.update(gen=int(m[1]), dec_s=float(m[2]), tok_s=float(m[3]), hit=int(m[5]), pcie=int(m[7]), file=int(m[8]))
        m = DS.search(l)
        if m: d.update(passes=int(m[1]), tok_pass=float(m[2]))
        m = LK.search(l)
        if m: d.update(lk_acc=int(m[1]), lk_dr=int(m[2]), lk_pass=int(m[3]))
    return d
def gen(name, ids, n):
    t = time.time()
    proc.stdin.write(f'GEN {n} temperature=0 ' + ','.join(map(str, ids)) + '\n'); proc.stdin.flush()
    toks = []
    while True:
        ln = proc.stdout.readline()
        if not ln: sys.exit('engine died mid-request')
        if ln.startswith('T '): toks.append(int(ln[2:]))
        elif ln.startswith('DONE') or ln.startswith('ERR'): break
    wall = time.time() - t
    d = figures()
    rec = dict(name=name, n_prompt=len(ids), wall=round(wall, 2), tokens=toks, **d)
    results.append(rec)
    print(f"[{label}] {name:10s} prompt {len(ids):6d} (+{d.get('new', 0):5d} new) {d.get('prompt_s', 0):7.2f} s [experts {d.get('experts', 0):6.2f} dense {d.get('dense', 0):6.2f} "
          f"cache-open {d.get('copen', 0):4.2f} seed {d.get('seed', 0):4.2f} {d.get('path', ''):6s}]  decode {len(toks):4d} tok {d.get('tok_s', 0):5.2f} tok/s {d.get('tok_pass', 0):4.2f} tok/pass"
          + (f"  lookup {d['lk_acc']}/{d['lk_dr']} in {d['lk_pass']} passes" if 'lk_acc' in d else ''), flush=True)
    json.dump(results, open(f'{out_dir}/results.json', 'w'), indent=1)
    return toks
gen('warm', pc['U1'], 4)
conv = pc['L']
a = gen('long_read', conv, 40)
for name, piece in (('A1_+150', 'new150'), ('A2_+400', 'new400'), ('A3_+1000', 'new1000'), ('A4_+60', 'new60'), ('A5_+2500', 'new2500')):
    conv = conv + a + pc[piece]
    a = gen(name, conv, 40)
conv = conv + a + pc['W']
gen('bench_long', conv, 200)
proc.stdin.write('QUIT\n'); proc.stdin.flush()
try: proc.wait(timeout=120)
except subprocess.TimeoutExpired: proc.kill()
print(f'[{label}] done', flush=True)
