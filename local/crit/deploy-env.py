#!/usr/bin/env python3
# deploy-env.py CONFIG SUFFIX NOTE KEY=VALUE... - back the config up as CONFIG.SUFFIX, set/override env keys (existing keys are
# changed in place, new ones appended after STRATA_CHUNK_ASYNC), append NOTE to _comment, validate.  The binary install and the
# restart are done by the caller (deploy-win6.sh does the whole thing when no existing key must change).
import json, re, shutil, sys
p, suffix, note = sys.argv[1], sys.argv[2], sys.argv[3]
kv = dict(a.split('=', 1) for a in sys.argv[4:])
shutil.copy2(p, p + '.' + suffix)
s = open(p).read()
for k, v in kv.items():
    m = re.search(r'^  "%s": "[^"]*"' % re.escape(k), s, re.M)
    if m:
        s = s[:m.start()] + '  "%s": "%s"' % (k, v) + s[m.end():]
    else:
        last = '  "STRATA_CHUNK_ASYNC": "1"\n'; assert s.count(last) == 1
        s = s.replace(last, '  "STRATA_CHUNK_ASYNC": "1",\n  "%s": "%s"\n' % (k, v))
if note:
    m = re.search(r'^ "_comment": "(.*)",$', s, re.M); assert m
    s = s[:m.end(1)] + ' ' + note.replace('"', "'") + s[m.end(1):]
c = json.loads(s)
for k, v in kv.items(): assert c['env'][k] == v, k
open(p, 'w').write(s)
print('config edited:', ', '.join('%s=%s' % x for x in kv.items()), '| backup', p + '.' + suffix)
