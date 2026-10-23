#!/usr/bin/env python3
"""local/secprof/build.py - the security-work corpora for the residency splice (tools/v41/splice_b.py):

  prof_requests.jsonl  chat requests (max_tokens 1) whose prompts the service routes with STRATA_ROUTE_COUNTS on:
                       NIST Juliet C/C++ test cases, MITRE CWE entries (descriptions, mitigations, demonstrative
                       code), real C/C++/Python code, each under a security-review instruction
  gen_tasks.jsonl      find-and-fix tasks the model answers (thinking on) - the experts it uses WRITING analyses/fixes
  sec_eval.npy         held-out token sequences [N, L] (plain text) for KL / top-1 against the production model
  side_tasks.jsonl     held-out find-and-fix tasks for a side-by-side read of production vs splice answers
  manifest.json        which CWE classes / entries / files went where

Profiling and held-out material never overlap: Juliet and the CWE catalog are split by weakness id, real code by
project.  Public, defensive material only.
    python3 -I local/secprof/build.py --data /home/yuya/datasets/security-profile --out /home/yuya/datasets/security-profile/build
"""
import argparse
import hashlib
import json
import os
import random
import re
import xml.etree.ElementTree as ET

import numpy as np
from tokenizers import Tokenizer

TOK = "/home/yuya/datasets/ds41-tokenizer/tokenizer.json"
SYSTEM = "You are a senior application security engineer. Be precise and concrete."
REVIEW = ("Review the following code for security vulnerabilities. For each issue give the CWE, the root cause and "
          "the exact lines involved, then provide a corrected version.\n\n```{lang}\n{code}\n```")
EXPLAIN = "Summarize this weakness for a code reviewer: how to recognize it in code and how to fix it.\n\n{text}"


def held_out(key):   # a stable ~1/7 split, independent of directory order
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 7 == 3


def read(path, limit=24000):
    with open(path, "r", errors="replace") as f:
        return f.read(limit)


def juliet(root, rng):
    """{cwe_dir: [file texts]}: up to 3 files per class - the _01 baseline (bad + good in one file) and two others."""
    out = {}
    for d in sorted(os.listdir(root)):
        files = []
        for dp, _, fs in os.walk(os.path.join(root, d)):
            files += [os.path.join(dp, f) for f in fs if f.endswith((".c", ".cpp"))]
        if not files:
            continue
        files.sort()
        base = [f for f in files if re.search(r"_01\.(c|cpp)$", f)][:1]
        rest = [f for f in files if f not in base]
        pick = base + rng.sample(rest, min(2, len(rest)))
        out[d] = [(f, read(f)) for f in pick]
    return out


def text_of(el):
    return " ".join(" ".join(el.itertext()).split()) if el is not None else ""


def cwe_entries(xml_path):
    """[(id, text)]: name, description, extended description, mitigations, demonstrative examples (code kept as
    lines), observed examples."""
    root = ET.parse(xml_path).getroot()
    ns = {"c": re.match(r"\{.*\}", root.tag).group(0)[1:-1]}
    out = []
    for w in root.iterfind(".//c:Weakness", ns):
        if w.get("Status") == "Deprecated":
            continue
        parts = ["CWE-%s: %s" % (w.get("ID"), w.get("Name")), text_of(w.find("c:Description", ns)),
                 text_of(w.find("c:Extended_Description", ns))]
        for m in w.iterfind("c:Potential_Mitigations/c:Mitigation", ns):
            parts.append("Mitigation (%s): %s" % (text_of(m.find("c:Phase", ns)), text_of(m.find("c:Description", ns))))
        for ex in w.iterfind("c:Demonstrative_Examples/c:Demonstrative_Example", ns):
            for ch in ex:
                tag = ch.tag.split("}")[1]
                if tag == "Example_Code":
                    code = "\n".join(ln.rstrip() for ln in "".join(ch.itertext()).splitlines() if ln.strip())
                    parts.append("```%s\n%s\n```" % ((ch.get("Language") or "").lower(), code))
                elif tag in ("Intro_Text", "Body_Text"):
                    parts.append(text_of(ch))
        for ob in w.iterfind("c:Observed_Examples/c:Observed_Example", ns):
            parts.append("%s: %s" % (text_of(ob.find("c:Reference", ns)), text_of(ob.find("c:Description", ns))))
        out.append((int(w.get("ID")), "\n\n".join(p for p in parts if p)))
    return out


def code_files(roots, exts, limit_files, rng):
    files = []
    for r in roots:
        for dp, _, fs in os.walk(r):
            if any(s in dp for s in ("/test", "/.git", "/build", "/vendor", "/node_modules")):
                continue
            files += [os.path.join(dp, f) for f in fs if f.endswith(exts)]
    files.sort()
    return [(f, read(f, 40000)) for f in rng.sample(files, min(limit_files, len(files)))]


def lang(path):
    return {"c": "c", "h": "c", "cpp": "cpp", "cc": "cpp", "hpp": "cpp", "py": "python", "js": "javascript",
            "ts": "typescript"}.get(path.rsplit(".", 1)[-1], "")


def chunks(tok, text, max_tokens):
    ids = tok.encode(text, add_special_tokens=False).ids
    for i in range(0, len(ids), max_tokens):
        yield tok.decode(ids[i:i + max_tokens])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-tokens", type=int, default=6000)
    ap.add_argument("--eval-seqs", type=int, default=16)
    ap.add_argument("--eval-len", type=int, default=2048)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(1017)
    tok = Tokenizer.from_file(TOK)
    jl = juliet(os.path.join(a.data, "juliet/C/testcases"), rng)
    cw = cwe_entries(os.path.join(a.data, "cwe/cwec_v4.20.xml"))
    ll = "/home/yuya/llm-src/strata-ds41/third_party/llama.cpp"
    real_in = code_files([ll + "/ggml/src/ggml-cpu", ll + "/ggml/src"], (".c", ".h"), 40, rng) + \
        code_files(["/home/yuya/llm-src/strata-ds41/tools"], (".py",), 30, rng)
    real_out = code_files([ll + "/tools/server", ll + "/common"], (".cpp", ".h"), 25, rng) + \
        code_files([ll + "/tools/server/webui/src"], (".ts", ".js"), 10, rng)
    man = {"juliet_profile": [], "juliet_heldout": [], "cwe_profile": 0, "cwe_heldout": 0,
           "real_profile": [p for p, _ in real_in], "real_heldout": [p for p, _ in real_out]}

    prof, gen, side, ev_text = [], [], [], []
    for d, fl in jl.items():
        ho = held_out(d)
        man["juliet_heldout" if ho else "juliet_profile"].append(d)
        for i, (p, t) in enumerate(fl):
            req = REVIEW.format(lang=lang(p), code=t)
            if ho:
                ev_text.append(t)
                if i == 0 and len(side) < 8:
                    side.append({"id": os.path.basename(p), "messages": [{"role": "system", "content": SYSTEM},
                                                                        {"role": "user", "content": req}]})
            else:
                prof.append(req)
                if i == 0 and rng.random() < 0.12:
                    gen.append({"id": os.path.basename(p), "messages": [{"role": "system", "content": SYSTEM},
                                                                       {"role": "user", "content": req}]})
    for cid, t in cw:
        if held_out("CWE-%d" % cid):
            man["cwe_heldout"] += 1
            ev_text.append(t)
        else:
            man["cwe_profile"] += 1
            prof.append(EXPLAIN.format(text=t))
    for p, t in real_in:
        for c in chunks(tok, t, a.chunk_tokens):
            prof.append(REVIEW.format(lang=lang(p), code=c))
    for p, t in real_out:
        ev_text.append(t)

    # profiling requests: pack short items into prompts of up to ~chunk_tokens
    rng.shuffle(prof)
    reqs, cur, cur_n = [], [], 0
    for t in prof:
        n = len(tok.encode(t, add_special_tokens=False).ids)
        if n > a.chunk_tokens:
            t = next(chunks(tok, t, a.chunk_tokens))
            n = a.chunk_tokens
        if cur and cur_n + n > a.chunk_tokens:
            reqs.append("\n\n---\n\n".join(cur))
            cur, cur_n = [], 0
        cur.append(t)
        cur_n += n
    if cur:
        reqs.append("\n\n---\n\n".join(cur))
    with open(os.path.join(a.out, "prof_requests.jsonl"), "w") as f:
        for r in reqs:
            f.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r}],
                                "max_tokens": 1}) + "\n")
    for name, rows in (("gen_tasks.jsonl", gen), ("side_tasks.jsonl", side)):
        with open(os.path.join(a.out, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # held-out eval sequences: plain text, a fixed mix (code first), cut to eval_len tokens each
    rng.shuffle(ev_text)
    stream = tok.encode("\n\n".join(ev_text), add_special_tokens=False).ids
    need = a.eval_seqs * a.eval_len
    assert len(stream) >= need, (len(stream), need)
    starts = sorted(rng.sample(range(0, len(stream) - a.eval_len, a.eval_len), a.eval_seqs))
    ev = np.array([stream[s:s + a.eval_len] for s in starts], np.int32)
    np.save(os.path.join(a.out, "sec_eval.npy"), ev)
    man.update({"profile_requests": len(reqs), "profile_tokens": sum(len(tok.encode(r, add_special_tokens=False).ids)
                                                                     for r in reqs),
                "gen_tasks": len(gen), "side_tasks": len(side), "eval": list(ev.shape),
                "heldout_stream_tokens": len(stream)})
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in man.items()}))


if __name__ == "__main__":
    main()
