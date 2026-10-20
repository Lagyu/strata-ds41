#!/usr/bin/env python3
"""unsloth_meta_to_ds41.py UNSLOTH_SHARD1 DS41_GGUF OUT - the metadata shard of unsloth's split (general.*, tokenizer.*,
split.*) with every `deepseek41.*` key of a DwarfStar-layout DS41 GGUF added (the geometry reader wants that scheme:
hidden_size, moe_intermediate_size, compress_ratios, engram.*, config, ...; unsloth's llama.cpp-style keys are dropped).
Both files describe the same checkpoint (DeepSeek-V4.1-Flash-1017)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "llama.cpp", "gguf-py"))
from gguf import GGUFReader, GGUFWriter, GGUFValueType
src, ref, out = sys.argv[1:4]
r, d = GGUFReader(src), GGUFReader(ref)
w = GGUFWriter(out, "deepseek41", use_temp_file=False)
def put(k, f):
    v = f.contents()
    t = f.types[0]
    if t == GGUFValueType.ARRAY:   # keep the element type (engram primes / multipliers are 64-bit; the writer would guess int32)
        if f.types[-1] == GGUFValueType.STRING:
            v = [x if isinstance(x, str) else bytes(x).decode("utf-8", "replace") for x in v]
        w.add_key_value(k, v, GGUFValueType.ARRAY, sub_type=f.types[-1])
    else:
        w.add_key_value(k, v, t)
n_u = n_d = 0
for k, f in r.fields.items():
    if k.startswith("GGUF.") or k == "general.architecture" or k.startswith("deepseek41."): continue
    put(k, f); n_u += 1
for k, f in d.fields.items():
    if not k.startswith("deepseek41."): continue
    put(k, f); n_d += 1
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
w.close()
print(f"wrote {out}: {n_u} unsloth keys kept + {n_d} deepseek41.* keys from the DS41 file")
