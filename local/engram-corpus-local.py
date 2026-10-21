#!/usr/bin/env python3
"""local/engram-corpus-local.py - the token corpus for the Engram hot set (build-hip/ds41_engram_hot rank / eval) from
what this box has: Project Gutenberg books (English prose, cut into ~6K-token documents), NeelNanda/pile-10k when it
downloads (web, code, StackExchange, patents, ...), XeStrata's Japanese docs, this tree's docs and sources.  Writes
OUT/train.i32 and OUT/heldout.i32 (int32 ids, -1 between documents; every 10th document held out), the format
tools/v41/engram_corpus.py writes.  Tokenizer: the model's tokenizer.json (tokenizers, no transformers needed).

  local/engram-corpus-local.py --out /home/yuya/models/dsv41-1017-strata/engram-corpus [--pile pile-10k.parquet]
"""
import argparse, glob, os, re, sys
import numpy as np

TOK = "/home/yuya/datasets/ds41-tokenizer/tokenizer.json"


def gutenberg():
    for f in sorted(glob.glob("/home/yuya/datasets/gutenberg/*.txt")):
        s = open(f, errors="replace").read()
        a = s.find("*** START"); b = s.find("*** END")
        if a > 0: s = s[s.find("\n", a) + 1:]
        if b > 0: s = s[:b]
        yield "gutenberg", s


def files(pattern_dirs, exts, name):
    for d in pattern_dirs:
        for root, _, fs in os.walk(d):
            if "/.git" in root or "/build" in root or "third_party" in root or "node_modules" in root:
                continue
            for f in fs:
                if os.path.splitext(f)[1] in exts:
                    p = os.path.join(root, f)
                    try:
                        s = open(p, errors="replace").read()
                    except OSError:
                        continue
                    if 2000 < len(s) < 400000:
                        yield name, s


def pile(path):
    """NeelNanda/pile-10k as its parquet file (curl'd from the Hub; no `datasets` library here)."""
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(path)
    except Exception as e:  # noqa: BLE001
        print("pile-10k unavailable:", str(e)[:120], file=sys.stderr)
        return
    texts, metas = t.column("text").to_pylist(), t.column("meta").to_pylist()
    for text, meta in zip(texts, metas):
        name = (meta or {}).get("pile_set_name", "unknown") if isinstance(meta, dict) else "unknown"
        yield "pile_" + str(name).split(" ")[0].lower(), text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pile", default="", help="NeelNanda/pile-10k parquet file ('' = skip)")
    ap.add_argument("--doc-tokens", type=int, default=6144, help="cut long texts into documents of this many tokens")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    os.makedirs(a.out, exist_ok=True)
    sources = [gutenberg(),
               files(["/home/yuya/llm-src/XeStrata/docs"], {".md"}, "ja_docs"),
               files(["/home/yuya/llm-src/strata-ds41/docs", "/home/yuya/llm-src/strata-ds41/tools/v41",
                      "/home/yuya/llm-src/strata-ds41/src", "/home/yuya/llm-src/strata-ds41/tools/ds4"],
                     {".md", ".py", ".cpp", ".hpp", ".cu", ".sh", ".h"}, "code_docs")]
    if a.pile:
        sources.append(pile(a.pile))
    train, held = [], []
    counts = {}
    n_doc = 0
    for src in sources:
        for name, text in src:
            ids = tok.encode(text, add_special_tokens=False).ids
            for i in range(0, len(ids), a.doc_tokens):
                doc = ids[i:i + a.doc_tokens]
                if len(doc) < 64:
                    continue
                n_doc += 1
                (held if n_doc % 10 == 0 else train).extend(doc + [-1])
                c = counts.setdefault(name, [0, 0]); c[0] += 1; c[1] += len(doc)
    np.array(train, dtype=np.int32).tofile(os.path.join(a.out, "train.i32"))
    np.array(held, dtype=np.int32).tofile(os.path.join(a.out, "heldout.i32"))
    for k, (d, t) in sorted(counts.items()):
        print(f"{k:22s} {d:6d} docs {t:10d} tokens")
    print(f"train {len(train)} ints, heldout {len(held)} ints -> {a.out}")


if __name__ == "__main__":
    main()
