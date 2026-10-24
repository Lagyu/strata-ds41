#!/usr/bin/env python3
"""bench_payload.py OUT_DIR - prompts for the batch-window benchmark (bench.py): four unrelated 2500-token conversations
B1..B4 (>= STRATA_SERVE_PARK_MIN, so each is parked and resumed without a re-read) and a short warm-up prompt, tokenized
from PUBLIC Gutenberg texts (nothing from any session).  Each piece is an .i32 file."""
import json
import os
import sys

import numpy as np

R = os.environ.get('BENCH_TREE') or '/home/yuya/llm-src/strata-ds41'
sys.path.insert(0, R)
sys.path.insert(0, f'{R}/tools')
sys.path.insert(0, f'{R}/tools/v41')
import strata_tokenizer as ST  # noqa: E402

TOKDIR = '/home/yuya/models/dsv41-1017-tokenizer'
BOOKS = '/home/yuya/datasets/gutenberg'
out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def load_tokenizer():
    vocab = json.load(open(os.path.join(TOKDIR, 'vocab.json'), encoding='utf-8'))
    tokens = [''] * (max(vocab.values()) + 1)
    for t, i in vocab.items():
        tokens[i] = t
    merges = [l.rstrip('\n') for l in open(os.path.join(TOKDIR, 'merges.txt'), encoding='utf-8') if not l.startswith('#version')]
    types = json.load(open(os.path.join(TOKDIR, 'token_type.json')))
    types = types if isinstance(types, list) else [types[str(i)] for i in range(len(tokens))]
    return ST.Tokenizer(tokens, merges, types, 'joyai-llm', {})


tok = load_tokenizer()
texts = {b: open(f'{BOOKS}/{b}.txt', errors='replace').read() for b in ('135', '1400', '158', '1342')}


def piece(book, char_off, n_tokens):
    ids = tok.encode(texts[book][char_off:char_off + n_tokens * 6], parse_special=False)
    assert len(ids) >= n_tokens, (book, char_off, len(ids))
    return ids[:n_tokens]


pieces = {
    'WARM': piece('1400', 1000000, 300),
    'B1': piece('135', 400000, 2500),
    'B2': piece('1400', 200000, 2500),
    'B3': piece('158', 200000, 2500),
    'B4': piece('1342', 200000, 2500),
}
for k, v in pieces.items():
    np.asarray(v, np.int32).tofile(f'{out}/{k}.i32')
    print(f'{k:6s} {len(v):5d} tokens')
json.dump({k: len(v) for k, v in pieces.items()}, open(f'{out}/sizes.json', 'w'), indent=1)
