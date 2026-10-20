#!/usr/bin/env python3
"""unsloth_dense_to_ds41.py IN_SHARD OUT_SHARD - rewrite unsloth's dense shard (…-00009-of-00009.gguf of
DeepSeek-V4.1-Flash-1017-GGUF UD-Q8_K_XL / UD-Q4_K_XL) into the types the DS41 engine's readers expect where they
read the bytes themselves: token_embd.weight -> F16 (the dense half and DSpark read its rows off the mmap), every
BF16 matrix -> F16 (the dense graph multiplies F16; BF16 -> F16 is exact for these FP8-derived weights - the script
counts the elements where it is not), hc_*_fn F32 -> F16 (as DwarfStar stores them).  Everything else is copied byte
for byte (Q8_0 stays Q8_0, F32 vectors stay F32).  The split metadata is kept.  Streams one tensor at a time.

  python3 tools/v41/unsloth_dense_to_ds41.py IN.gguf OUT.gguf
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "llama.cpp", "gguf-py"))
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
w = GGUFWriter(dst, "deepseek41", use_temp_file=False)
n_gates = sum(1 for t in r.tensors if t.name.endswith(".engram_gate.weight"))
for k, f in r.fields.items():
    if k.startswith("split."):
        v = int(f.parts[f.data[0]][0])
        if k == "split.tensors.count": v += n_gates   # each engram_gate becomes two tensors (q_norm, k_norm)
        (w.add_uint16 if k != "split.tensors.count" else w.add_int32)(k, v)

def ne(t):            # GGUF order [ne0, ne1, ...]
    return [int(x) for x in t.shape.tolist()]
def to_f16(t):        # the tensor's values as a float16 array in numpy order (ne1, ne0)
    raw = np.asarray(t.data)
    n0, n1 = ne(t)[0], int(t.n_elements) // ne(t)[0]
    ty = t.tensor_type
    if ty == T.BF16:
        u16 = raw.reshape(-1).view(np.uint16).reshape(n1, n0)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    elif ty == T.F32:
        f32 = raw.reshape(-1).view(np.float32).reshape(n1, n0)
    elif ty == T.F16:
        f32 = raw.reshape(-1).view(np.float16).reshape(n1, n0).astype(np.float32)
    elif ty == T.Q8_0:   # block: f16 scale + 32 int8 (34 bytes); row = ne0/32 blocks
        b = raw.reshape(-1).view(np.uint8).reshape(n1, n0 // 32, 34)
        scale = b[:, :, :2].copy().view(np.float16).astype(np.float32).reshape(n1, n0 // 32, 1)
        q = b[:, :, 2:].view(np.int8).astype(np.float32)
        f32 = (q * scale).reshape(n1, n0)
    else:
        raise SystemExit(f"{t.name}: cannot convert {ty.name} to F16")
    f16 = f32.astype(np.float16)
    return f16, int(np.count_nonzero(f16.astype(np.float32) != f32))

def converts(t):
    return t.name == "token_embd.weight" or t.tensor_type == T.BF16 or \
        (t.tensor_type == T.F32 and (t.name.endswith("hc_attn_fn.weight") or t.name.endswith("hc_ffn_fn.weight")))

# unsloth's blk.L.engram_gate.weight [D, HC] is exactly DwarfStar's engram_q_norm * engram_k_norm (checked: max diff 0 on
# both Engram layers), which the dense graph multiplies at run time: it becomes engram_q_norm, with engram_k_norm = ones,
# so the product is the same bits; the gate tensor itself is not written
gates = [t for t in r.tensors if t.name.endswith(".engram_gate.weight")]
synth = {}   # name -> float32 array in numpy order
for t in gates:
    g = np.asarray(t.data).reshape(-1).view(np.float32).reshape(int(t.n_elements) // int(t.shape[0]), int(t.shape[0]))
    base = t.name[: -len("engram_gate.weight")]
    synth[base + "engram_q_norm.weight"] = g.copy()
    synth[base + "engram_k_norm.weight"] = np.ones_like(g)
# pass 1: the tensor infos (numpy-order shapes, as GGUFWriter.add_tensor passes them)
for name, arr in synth.items():
    w.add_tensor_info(name, list(arr.shape), np.dtype(np.float32), arr.nbytes, raw_dtype=T.F32)
for t in r.tensors:
    if t.name.endswith(".engram_gate.weight"): continue
    shape_np = list(reversed(ne(t)))
    if converts(t):
        w.add_tensor_info(t.name, shape_np, np.dtype(np.float16), int(t.n_elements) * 2, raw_dtype=T.F16)
    else:   # a uint8 tensor's shape is its BYTE shape (the writer turns the last dim back into elements)
        bshape = shape_np[:-1] + [int(t.n_bytes) // max(1, int(np.prod(shape_np[:-1])))]
        w.add_tensor_info(t.name, bshape, np.dtype(np.uint8), int(t.n_bytes), raw_dtype=t.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
# pass 2: the bytes, one tensor at a time
n_conv = n_copy = lossy = 0
for name, arr in synth.items():
    w.write_tensor_data(np.ascontiguousarray(arr))
for t in r.tensors:
    if t.name.endswith(".engram_gate.weight"): continue
    if converts(t):
        f16, bad = to_f16(t)
        w.write_tensor_data(np.ascontiguousarray(f16))
        n_conv += 1; lossy += bad
    else:
        w.write_tensor_data(np.ascontiguousarray(np.asarray(t.data).reshape(-1).view(np.uint8)))
        n_copy += 1
w.close()
print(f"wrote {dst}: converted to F16 x{n_conv}, copied x{n_copy}, synthesized {len(synth)} Engram norms from {len(gates)} gates; elements changed by the F16 rounding: {lossy}")
