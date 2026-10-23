#!/usr/bin/env python3
"""Drive one `ds41_generate --serve` over its line protocol: the shared-prefix parks, the whole-conversation parks and
the request slots (BGEN / BT / BDONE / BSTOP / BADM).

  run.py OUT_DIR PAYLOAD_DIR {prefix|endpark|slots} [ENV=VALUE ...]

prefix:  P alone; D = P + tailC as a continuation (reference logits); an unrelated prompt; B = P + tailB read in full
         (shares P with D's park -> the read is split and P becomes a shared-prefix park); another unrelated prompt
         (PARK_SLOTS=2 evicts D's conversation, never the prefix); C = P + tailC again -> must come from the prefix park
         (RESUME 6000) with logits IDENTICAL to D's; E = P + tailD (prefix hit too).
endpark: W decodes 40 tokens; W2 = W + those tokens + tailW continues the live conversation (reference logits); an
         unrelated prompt parks it; W again from scratch (same batch shape) decodes 40 tokens again; an unrelated prompt
         parks THAT conversation at its END; W2 again -> must resume at the END (RESUME 3040 or 3039) with the same logits.
slots:   solo S1 / S2 (30 greedy tokens each, the references); BGEN 0 S1 + BGEN 1 S2 at once (slice switching, BSTOP 1
         after 10 tokens); then BGEN 0 S1 and BGEN 1 S2 one after the other (no switching): the token streams must agree;
         GEN S3 stopped after 8 tokens and continued as BGEN 0 (the server's promotion) vs a plain GEN S3.
Every request's last-row logits are dumped by the engine (STRATA_SERVE_DUMP); results.json has what the engine said.
Engine command, arguments and environment are the production config's (local/ds41-yuya-3gpu.json)."""
import json
import os
import subprocess
import sys
import time

import numpy as np

out_dir, payload, mode = sys.argv[1], sys.argv[2], sys.argv[3]
extra_env = dict(kv.split('=', 1) for kv in sys.argv[4:])
pin_batch = extra_env.pop('PROMPT_BATCH', None)
R = '/home/yuya/llm-src/strata-ds41'
cfg = json.load(open(f'{R}/local/ds41-yuya-3gpu.json'))
os.makedirs(f'{out_dir}/dump', exist_ok=True)
pc = {k: np.fromfile(f'{payload}/{k}.i32', np.int32).tolist() for k in json.load(open(f'{payload}/sizes.json'))}

env = dict(os.environ)
env['LD_LIBRARY_PATH'] = '/opt/rocm/lib' + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
env.update({k: str(v) for k, v in cfg['env'].items() if k != 'STRATA_ROUTE_COUNTS'})   # tests must not write the production counts
env.update(STRATA_SERVE_DUMP=f'{out_dir}/dump')
env.update(extra_env)
args = list(cfg['args'])
if pin_batch:
    args[args.index('--prompt-batch') + 1] = pin_batch
cmd = [f'{R}/local/ds41-engine.sh', '--serve'] + args
log = open(f'{out_dir}/engine.log', 'w')
proc = subprocess.Popen(cmd, cwd=R, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
t0 = time.time()
info_lines = []
while True:
    ln = proc.stdout.readline()
    if not ln:
        sys.exit('engine died before READY (see engine.log)')
    if ln.startswith('INFO'):
        info_lines.append(ln.strip())
    if ln.startswith('READY'):
        break
print(f'ready after {time.time() - t0:.0f} s: {ln.strip()}  {" | ".join(info_lines)}', flush=True)

results = []
n_done = 0      # the engine's dump index: one per prompt read that completed (DONE from GEN, DONE from a BGEN admission)
log_pos = 0


def log_lines():
    """the engine's decision lines written since the last call"""
    global log_pos
    log.flush()
    with open(f'{out_dir}/engine.log', 'rb') as f:
        f.seek(log_pos)
        text = f.read().decode('utf-8', 'replace')
        log_pos = f.tell()
    return [l for l in text.splitlines() if l.startswith(('serve: reuse', 'serve: unparked', 'serve: parked', 'serve: shared prefix',
                                                            'serve: tail resume', 'serve: park full', 'serve: prefix slots'))]


def send(s):
    proc.stdin.write(s + '\n')
    proc.stdin.flush()


def readline():
    ln = proc.stdout.readline()
    if not ln:
        sys.exit('engine died mid-request')
    return ln.strip()


def gen(name, ids, n, stop_after=None, expect=None):
    """a solo request (GEN): tokens, RESUME, DONE"""
    global n_done
    idx = n_done
    t = time.time()
    send(f'GEN {n} temperature=0 ' + ','.join(map(str, ids)))
    toks, info, stopped = [], {}, False
    while True:
        ln = readline()
        if ln.startswith('T '):
            toks.append(int(ln[2:]))
            if stop_after and len(toks) == stop_after and not stopped:
                send('STOP'); stopped = True
        elif ln.startswith('RESUME'):
            info['resume'] = int(ln.split()[1])
        elif ln.startswith('DONE') or ln.startswith('ERR'):
            info['end'] = ln
            break
    info['wall'] = round(time.time() - t, 1)
    n_done += info['end'].startswith('DONE')
    rec = dict(name=name, kind='GEN', dump_index=idx, n_prompt=len(ids), expect=expect, tokens=toks, log=log_lines(), **info)
    results.append(rec)
    print(f"{name:14s} GEN    prompt {len(ids):6d} expect {str(expect):9s} resume {info.get('resume', 0):6d} wall {info['wall']:6.1f} s  "
          f"tokens {len(toks)} {toks[:5]}  {info['end'][:40]}", flush=True)
    for l in rec['log']:
        print('      ' + l[:150], flush=True)
    json.dump(results, open(f'{out_dir}/results.json', 'w'), indent=1)
    return rec


class Slot:
    def __init__(self, name, slot, ids, n, expect=None, stop_after=None):
        self.name, self.slot, self.ids, self.n, self.expect, self.stop_after = name, slot, ids, n, expect, stop_after
        self.toks, self.info, self.stopped, self.done = [], {}, False, False
        self.t0 = time.time()
        self.first_at = None


def admit(sl, others):
    """send BGEN for sl and read its admission (PP / RESUME / T / DONE / BADM), passing BT / BDONE lines of the other slots on"""
    global n_done
    sl.dump_index = n_done
    send(f'BGEN {sl.slot} {sl.n} temperature=0 ' + ','.join(map(str, sl.ids)))
    while True:
        ln = readline()
        if ln.startswith('BT ') or ln.startswith('BDONE '):
            slot_line(ln, others)
        elif ln.startswith('T '):
            sl.toks.append(int(ln[2:])); sl.first_at = time.time()
        elif ln.startswith('RESUME'):
            sl.info['resume'] = int(ln.split()[1])
        elif ln.startswith('DONE'):
            sl.info['adm'] = ln
            n_done += 1
        elif ln.startswith('BADM'):
            sl.info['badm'] = ln
            if ln.split()[2] == '0':
                sl.done = True
                sl.info['end'] = 'ended at admission: ' + sl.info.get('adm', '')
            break
        elif ln.startswith('ERR'):
            sl.info['end'] = ln
            sl.done = True
            break
    for l in log_lines():
        sl.info.setdefault('log', []).append(l)
    print(f"{sl.name:14s} BGEN {sl.slot} prompt {len(sl.ids):6d} expect {str(sl.expect):9s} resume {sl.info.get('resume', 0):6d} "
          f"admitted in {time.time() - sl.t0:5.1f} s  {sl.info.get('badm', sl.info.get('end', ''))}", flush=True)
    for l in sl.info.get('log', []):
        print('      ' + l[:150], flush=True)


def slot_line(ln, slots):
    f = ln.split()
    if f[0] not in ('BT', 'BDONE'):   # INFO / PP lines of the engine between passes: not a slot's
        return
    s = int(f[1])
    sl = next((x for x in slots if x.slot == s and not x.done), None)
    if sl is None:
        print('      stray line ' + ln, flush=True)
        return
    if f[0] == 'BT':
        sl.toks.append(int(f[2]))
        if sl.stop_after and len(sl.toks) >= sl.stop_after and not sl.stopped:
            send(f'BSTOP {s}'); sl.stopped = True
    else:
        sl.info['end'] = ln
        sl.done = True
        sl.wall = round(time.time() - sl.t0, 1)


def run_slots(slots):
    """the slots are admitted in order; every BT/BDONE line goes to its slot until all are done"""
    for i, sl in enumerate(slots):
        admit(sl, slots)
    while any(not s.done for s in slots):
        slot_line(readline(), slots)
    for sl in slots:
        rec = dict(name=sl.name, kind=f'BGEN{sl.slot}', dump_index=sl.dump_index, n_prompt=len(sl.ids), expect=sl.expect,
                   tokens=sl.toks, wall=getattr(sl, 'wall', None), **sl.info)
        results.append(rec)
        print(f"{sl.name:14s} slot {sl.slot} done: {len(sl.toks)} tokens {sl.toks[:5]}  {sl.info.get('end', '')[:40]}", flush=True)
    json.dump(results, open(f'{out_dir}/results.json', 'w'), indent=1)


P = pc['P']
if mode == 'prefix':
    gen('P', P, 1)
    gen('D_direct', P + pc['tailC'], 1, expect=len(P))
    gen('U1', pc['U1'], 1)
    gen('B_split', P + pc['tailB'], 1, expect=0)                 # read in full, split at 6000: the prefix park
    gen('U2', pc['U2'], 1)                                        # PARK_SLOTS=2: the ordinary parks come and go, the prefix stays
    gen('U3', pc['U3'], 1)                                        # (B's own conversation is evicted here)
    gen('C_prefix', P + pc['tailB'], 1, expect=len(P))           # B's tail again, from the prefix park: logits == B_split's
    gen('U4', pc['U4'], 1)
    gen('E_prefix', P + pc['tailD'], 1, expect=len(P))           # a second request on the prefix
    gen('P_regen', P, 1, expect=len(P))                           # the prefix itself: its saved logits, no read
elif mode == 'endpark':
    W = pc['W']
    w60 = gen('W60', W, 60)                                       # the uninterrupted reference
    ans = w60['tokens']
    gen('U1', pc['U1'], 1)                                        # parks W60's conversation (prompt end 3000, END 3060)
    w40 = gen('W40', W, 40, expect=len(W))                        # the same prompt: regen from the park's prompt-end state
    print(f"      W40 == W60[:40]: {w40['tokens'] == ans[:40]}", flush=True)
    gen('U2', pc['U2'], 1)                                        # parks W40's conversation WITH its answer (END rings)
    c = gen('W_cont', W + w40['tokens'] + [ans[40]], 19, expect=('end', len(W) + 40))   # continues at the END + 1 new token
    gen('U3', pc['U3'], 1)                                        # parks W_cont's conversation
    e_end = len(W) + 41 + len(c['tokens']) - 1
    x = pc['tailW'][0] if pc['tailW'][0] != ans[55] else pc['tailW'][1]
    gen('W_tail', W + w40['tokens'] + ans[40:55] + [x], 4, expect=('tail', e_end))   # diverges within 17 of that END: tail resume
elif mode == 'prefix2':
    # a second engine start with STRATA_SERVE_PREFIX_DIR: the prefix saved by the `prefix` run must be loaded and hit
    gen('U5', pc['U5'], 1)
    gen('C_loaded', P + pc['tailC'], 1, expect=len(P))           # from the loaded prefix file: RESUME 6000, no read of P
    gen('P_regen2', P, 1, expect=len(P))
elif mode == 'cancel':
    # Stop during a long read (Maya v1.1.20's rule): the read ends within a layer, the part read stays loaded, the resend
    # continues from there
    gen('U1', pc['U1'], 1)
    import threading
    def stopper():
        time.sleep(float(os.environ.get('CANCEL_AFTER_S', '9')))
        send('STOP')
    threading.Thread(target=stopper, daemon=True).start()
    c = gen('long_cut', P + pc['tailB'] + pc['tailC'], 8)        # 7400 tokens, ~32 s to read: STOP after ~9 s
    gen('long_resend', P + pc['tailB'] + pc['tailC'] + pc['tailD'][:40], 4, expect='partial')   # continues from the part read
    gen('U2', pc['U2'], 1)
    gen('long_again', P + pc['tailB'] + pc['tailC'] + pc['tailD'][:40] + pc['tailD'][40:80], 4, expect='end')   # from the park at its END
elif mode == 'slots':
    S1, S2, S3 = pc['S1'], pc['S2'], pc['S3']
    gen('S1_solo', S1, 30)
    gen('S2_solo', S2, 30)
    gen('S3_solo', S3, 40)
    gen('U4', pc['U4'], 1)
    run_slots([Slot('S1_par', 0, S1, 30), Slot('S2_par', 1, S2, 30)])                    # switching every slice
    gen('U5', pc['U5'], 1)
    run_slots([Slot('S1_seq', 0, S1, 30)])                                               # one at a time
    run_slots([Slot('S2_seq', 1, S2, 30)])
    run_slots([Slot('S1_stop', 0, S1, 30), Slot('S2_stop', 1, S2, 30, stop_after=10)])  # BSTOP 1 after 10 tokens
    r3 = gen('S3_cut', S3, 40, stop_after=8)                                             # the server's promotion: STOP, then
    run_slots([Slot('S3_cont', 0, S3 + r3['tokens'], 40 - len(r3['tokens']), expect=len(S3) + len(r3['tokens']) - 1)])   # BGEN with the output so far
send('QUIT')
try:
    proc.wait(timeout=120)
except subprocess.TimeoutExpired:
    proc.kill()
print('done', flush=True)
