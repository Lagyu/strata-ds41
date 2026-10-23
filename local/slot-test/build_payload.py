#!/usr/bin/env python3
"""build_payload.py OUT_DIR - synthetic prompts for the shared-prefix / whole-conversation park / request-slot tests of
`ds41_generate --serve`, tokenized from PUBLIC Gutenberg texts (nothing from any session).  Each piece is an .i32 file."""
import json
import os
import sys

import numpy as np

R = '/home/yuya/llm-src/strata-ds41'
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
    'P': piece('135', 200000, 6000),        # the shared prefix
    'tailB': piece('135', 900000, 600),
    'tailC': piece('135', 1300000, 800),
    'tailD': piece('135', 1700000, 800),
    'U1': piece('1400', 100000, 2500),      # unrelated conversations (>= STRATA_SERVE_PARK_MIN, so they get parked)
    'U2': piece('1400', 400000, 2500),
    'U3': piece('1400', 700000, 2500),
    'U4': piece('158', 100000, 2500),
    'U5': piece('158', 400000, 2500),
    'W': piece('1342', 100000, 3000),
    'tailW': piece('1342', 500000, 100),
    'S1': piece('158', 600000, 2500),
    'S2': piece('1342', 300000, 2500),
    'S3': piece('135', 2200000, 2500),
}
for k, v in pieces.items():
    np.asarray(v, np.int32).tofile(f'{out}/{k}.i32')
    print(f'{k:6s} {len(v):5d} tokens')
json.dump({k: len(v) for k, v in pieces.items()}, open(f'{out}/sizes.json', 'w'), indent=1)
