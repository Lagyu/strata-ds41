#!/usr/bin/env python3
"""build_payload.py OUT_DIR - synthetic prompts for the performance probe (public Gutenberg text and this tree's own code,
nothing from any session).  Pieces are .i32 token files; `copy_task.i32` is a rendered chat prompt (thinking off) that asks
the model to reproduce a code file with a small change - the lookup-decoding case (an agent's output copies its context)."""
import json
import os
import sys

import numpy as np

R = '/home/yuya/llm-src/strata-ds41'
sys.path.insert(0, R)
sys.path.insert(0, f'{R}/tools')
sys.path.insert(0, f'{R}/tools/v41')
import strata_tokenizer as ST  # noqa: E402
import dsv41_encoding as E  # noqa: E402

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


code = open(f'{R}/serve/cache_drop.py', encoding='utf-8').read()
copy_msgs = [{'role': 'user', 'content': 'Below is a Python file. Output the complete file again, unchanged except that every '
              'comment line (a line whose first non-blank character is #) is removed. Output only the code, nothing else.\n\n'
              '```python\n' + code + '\n```'}]
copy_ids = tok.encode(E.encode_messages(copy_msgs, thinking_mode='chat'), parse_special=True)
pieces = {
    'P': piece('135', 200000, 6000),
    'tailB': piece('135', 900000, 600),
    'new150': piece('1400', 100000, 150),
    'new400': piece('1400', 300000, 400),
    'new1000': piece('1400', 500000, 1000),
    'new60': piece('1400', 800000, 60),
    'new2500': piece('1400', 900000, 2500),
    'U1': piece('158', 100000, 2500),
    'W': piece('1342', 100000, 3000),
    'S1': piece('158', 600000, 2500),
    'copy_task': copy_ids,
}
for k, v in pieces.items():
    np.asarray(v, np.int32).tofile(f'{out}/{k}.i32')
    print(f'{k:10s} {len(v):5d} tokens')
json.dump({k: len(v) for k, v in pieces.items()}, open(f'{out}/sizes.json', 'w'), indent=1)
