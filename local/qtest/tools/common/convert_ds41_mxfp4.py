#!/usr/bin/env python3
"""Build a Lucebox DeepSeek-V4.1 GGUF whose routed experts are the checkpoint's own MXFP4 blocks.

  convert_ds41_mxfp4.py build  --src LUCEBOX.gguf --shards DIR --out NEW.gguf [--wait] [--workers 12]
  convert_ds41_mxfp4.py check  --src LUCEBOX.gguf --shards DIR --out NEW.gguf [--layer 0 --expert 0]

build: the metadata and every non-expert tensor (attention, shared experts, norms, head, hyper-connections;
Q8_0 with the abliteration already applied) are copied from the source Lucebox file; the three expert
tensors of each layer (blk.N.ffn_{gate,up,down}_exps.weight) are repacked from the HF shards' MXFP4
(I8 codes [rows, cols/2], element 2i low nibble, 2i+1 high; F8_E8M0 scales [rows, cols/32]) into ggml's
block_mxfp4 (scale byte, then 16 bytes: element j low, j+16 high) - a bit move, no requantization.
The Engram tables are left out (LUCE_DS4_ENGRAM_GGUF points Lucebox at the source file for them).
--wait: a layer is converted as soon as its shards are complete (no .aria2 beside them).

check: dequantizes one expert's gate matrix three ways - ggml's dequantize_row_mxfp4 on the new file,
the checkpoint's formula on the shard bytes, and ggml's ROCMFP2 dequantizer on the source file - and
reports exact equality of the first two and the correlation with the third (a wrong row/nibble order
shows up as ~0 correlation).
"""
import argparse
import ctypes
import json
import mmap
import multiprocessing as mp
import os
import struct
import sys
import time

import numpy as np

ALIGN = 16384
GGML_TYPE_MXFP4 = 39
QK = 32
LIBGGML = "/home/yuya/llm-src/lucebox/server/build-hip/deps/llama.cpp/ggml/src/libggml-base.so"
EXPERT_RE = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
PROJ = {"ffn_gate_exps.weight": "w1", "ffn_up_exps.weight": "w3", "ffn_down_exps.weight": "w2"}

# ---------------------------------------------------------------- GGUF header read/write
KV_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Reader:
    def __init__(self, f):
        self.f = f

    def rd(self, fmt):
        return struct.unpack("<" + fmt, self.f.read(struct.calcsize(fmt)))

    def rstr(self):
        (n,) = self.rd("Q")
        return self.f.read(n).decode("utf-8")

    def rval(self, t):
        if t == 8:
            return self.rstr()
        if t == 9:
            (et,) = self.rd("I")
            (n,) = self.rd("Q")
            if et == 8:
                return (et, [self.rstr() for _ in range(n)])
            fmt = KV_FMT[et]
            vals = list(struct.unpack("<%d%s" % (n, fmt), self.f.read(n * struct.calcsize(fmt))))
            return (et, vals)
        (v,) = self.rd(KV_FMT[t])
        return v


def read_header(path):
    with open(path, "rb") as f:
        r = Reader(f)
        if f.read(4) != b"GGUF":
            raise SystemExit(path + ": not a GGUF")
        (ver,) = r.rd("I")
        (nt,) = r.rd("Q")
        (nkv,) = r.rd("Q")
        kvs = []
        align = 32
        for _ in range(nkv):
            k = r.rstr()
            (t,) = r.rd("I")
            v = r.rval(t)
            kvs.append((k, t, v))
            if k == "general.alignment":
                align = v
        tensors = []
        for _ in range(nt):
            name = r.rstr()
            (nd,) = r.rd("I")
            dims = list(r.rd("Q" * nd))
            (ty,) = r.rd("I")
            (off,) = r.rd("Q")
            tensors.append({"name": name, "dims": dims, "type": ty, "offset": off})
        header_end = f.tell()
    data_start = (header_end + align - 1) // align * align
    for i, t in enumerate(tensors):
        t["abs"] = data_start + t["offset"]
    return ver, kvs, tensors, align, data_start


def enc_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def enc_val(t, v):
    if t == 8:
        return enc_str(v)
    if t == 9:
        et, vals = v
        out = struct.pack("<IQ", et, len(vals))
        if et == 8:
            return out + b"".join(enc_str(x) for x in vals)
        return out + struct.pack("<%d%s" % (len(vals), KV_FMT[et]), *vals)
    return struct.pack("<" + KV_FMT[t], v)


def write_header(f, ver, kvs, tensors, align):
    f.write(b"GGUF" + struct.pack("<IQQ", ver, len(tensors), len(kvs)))
    for k, t, v in kvs:
        f.write(enc_str(k) + struct.pack("<I", t) + enc_val(t, v))
    for t in tensors:
        f.write(enc_str(t["name"]) + struct.pack("<I", len(t["dims"])) + struct.pack("<%dQ" % len(t["dims"]), *t["dims"])
                + struct.pack("<IQ", t["type"], t["offset"]))
    end = f.tell()
    pad = (end + align - 1) // align * align - end
    f.write(b"\0" * pad)
    return end + pad


# ---------------------------------------------------------------- ggml type sizes via libggml
def ggml():
    L = ctypes.CDLL(LIBGGML)
    L.ggml_row_size.restype = ctypes.c_size_t
    L.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    L.ggml_blck_size.restype = ctypes.c_int64
    return L


def tensor_bytes(L, t):
    dims = t["dims"]
    n = L.ggml_row_size(t["type"], dims[0])
    for d in dims[1:]:
        n *= d
    return n


# ---------------------------------------------------------------- shards
def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


class Shards:
    def __init__(self, d):
        self.dir = d
        self.index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.headers = {}

    def shard_of(self, name):
        return self.index[name]

    def complete(self, shard):
        p = os.path.join(self.dir, shard)
        return os.path.exists(p) and not os.path.exists(p + ".aria2") and os.path.getsize(p) > 8

    def locate(self, name):
        shard = self.index[name]
        if shard not in self.headers:
            self.headers[shard] = st_header(os.path.join(self.dir, shard))
        h, ds = self.headers[shard]
        t = h[name]
        a, b = t["data_offsets"]
        return os.path.join(self.dir, shard), ds + a, b - a, t["dtype"], t["shape"]


# ---------------------------------------------------------------- the repack (llama.cpp's repack_mxfp4_blocks)
def repack(packed, scale):
    """packed uint8 [rows, cols/2] (2i low nibble, 2i+1 high), scale uint8 [rows, cols/32] -> [rows, cols/32*17]."""
    rows, pc = packed.shape
    nb = pc * 2 // QK
    src = packed.reshape(rows, nb, 16)
    lo = src & 0x0F
    hi = src >> 4
    vals = np.empty((rows, nb, 32), dtype=np.uint8)
    vals[:, :, 0::2] = lo
    vals[:, :, 1::2] = hi
    qs = vals[:, :, :16] | (vals[:, :, 16:] << 4)
    out = np.empty((rows, nb, 17), dtype=np.uint8)
    out[:, :, 0] = scale.reshape(rows, nb)
    out[:, :, 1:] = qs
    return out.reshape(rows, nb * 17)


def expert_job(args):
    """One expert of one tensor: read its shard bytes, repack, pwrite into the output at its slot."""
    shard_path, w_off, w_len, s_off, s_len, rows, cols, out_path, out_off = args
    with open(shard_path, "rb") as f:
        f.seek(w_off)
        w = np.frombuffer(f.read(w_len), dtype=np.uint8).reshape(rows, cols // 2)
        f.seek(s_off)
        s = np.frombuffer(f.read(s_len), dtype=np.uint8).reshape(rows, cols // QK)
    blob = repack(w, s).tobytes()
    fd = os.open(out_path, os.O_WRONLY)
    try:
        n = os.pwrite(fd, blob, out_off)
    finally:
        os.close(fd)
    return n


def copy_range(src, dst, src_off, dst_off, n, chunk=64 << 20):
    done = 0
    while done < n:
        b = os.pread(src, min(chunk, n - done), src_off + done)
        if not b:
            raise RuntimeError("short read copying %d bytes at %d" % (n, src_off))
        os.pwrite(dst, b, dst_off + done)
        done += len(b)


# ---------------------------------------------------------------- build
def build(a):
    L = ggml()
    ver, kvs, src_tensors, align, _ = read_header(a.src)
    shards = Shards(a.shards)
    # the new tensor list
    out_tensors = []
    off = 0
    for t in src_tensors:
        if ".engram_embd." in t["name"]:
            continue
        nt = {"name": t["name"], "dims": list(t["dims"]), "type": t["type"], "src": t}
        if any(t["name"].endswith(e) for e in EXPERT_RE):
            nt["type"] = GGML_TYPE_MXFP4
        nt["bytes"] = tensor_bytes(L, nt)
        off = (off + align - 1) // align * align
        nt["offset"] = off
        off += nt["bytes"]
        out_tensors.append(nt)
    total_data = off
    kvs = [(k, t, ("MXFP4 gate/up/down (the checkpoint's blocks); Q8_0 attention/shared/head" if k == "deepseek41.quantization" else v))
           for k, t, v in kvs]
    n_exp = sum(1 for t in out_tensors if t["type"] == GGML_TYPE_MXFP4)
    print(f"{len(out_tensors)} tensors ({n_exp} expert stacks as MXFP4), data {total_data / 1e9:.1f} GB", flush=True)
    with open(a.out, "wb") as f:
        data_start = write_header(f, ver, kvs, out_tensors, align)
    os.truncate(a.out, data_start + total_data)
    print(f"header written, data starts at {data_start}, file size {(data_start + total_data) / 1e9:.1f} GB", flush=True)

    sfd = os.open(a.src, os.O_RDONLY)
    dfd = os.open(a.out, os.O_WRONLY)
    t0 = time.time()
    copied = 0
    for t in out_tensors:
        if t["type"] == GGML_TYPE_MXFP4:
            continue
        copy_range(sfd, dfd, t["src"]["abs"], data_start + t["offset"], t["bytes"])
        copied += t["bytes"]
    os.close(sfd)
    os.close(dfd)
    print(f"copied {copied / 1e9:.2f} GB of non-expert tensors in {time.time() - t0:.0f} s", flush=True)

    layers = {}
    for t in out_tensors:
        if t["type"] != GGML_TYPE_MXFP4:
            continue
        il = int(t["name"].split(".")[1])
        layers.setdefault(il, []).append(t)
    pool = mp.Pool(a.workers)
    for il in sorted(layers):
        names = [f"layers.{il}.ffn.experts.{e}.{p}.{k}" for e in range(384) for p in ("w1", "w2", "w3") for k in ("weight", "scale")]
        need = sorted(set(shards.shard_of(n) for n in names))
        while not all(shards.complete(s) for s in need):
            if not a.wait:
                raise SystemExit(f"layer {il}: shards {need} not complete (use --wait)")
            time.sleep(30)
        t1 = time.time()
        jobs = []
        for t in layers[il]:
            proj = PROJ[t["name"].split(".", 2)[2]]
            ne0, ne1, n_expert = t["dims"]
            per_expert = L.ggml_row_size(GGML_TYPE_MXFP4, ne0) * ne1
            for e in range(n_expert):
                wp, wo, wl, wd, ws = shards.locate(f"layers.{il}.ffn.experts.{e}.{proj}.weight")
                sp, so, sl, sd, ss = shards.locate(f"layers.{il}.ffn.experts.{e}.{proj}.scale")
                if wd != "I8" or sd != "F8_E8M0" or ws != [ne1, ne0 // 2] or ss != [ne1, ne0 // QK]:
                    raise SystemExit(f"layer {il} expert {e} {proj}: unexpected {wd} {ws} / {sd} {ss} for dims {t['dims']}")
                jobs.append((wp, wo, wl, so, sl, ne1, ne0, a.out, data_start + t["offset"] + e * per_expert))
        n = sum(pool.imap_unordered(expert_job, jobs, chunksize=8))
        print(f"layer {il:2d}: {len(jobs)} experts, {n / 1e9:.2f} GB in {time.time() - t1:.0f} s", flush=True)
    pool.close()
    pool.join()
    os.sync()
    print(f"done in {time.time() - t0:.0f} s: {a.out}", flush=True)


# ---------------------------------------------------------------- check
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def check(a):
    L = ggml()
    L.dequantize_row_mxfp4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    L.rocmfpx_dequantize_row_fp2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    _, _, src_tensors, _, _ = read_header(a.src)
    _, _, out_tensors, _, _ = read_header(a.out)
    shards = Shards(a.shards)
    proj = {"gate": "w1", "up": "w3", "down": "w2"}[a.which]
    name = f"blk.{a.layer}.ffn_{a.which}_exps.weight"
    st = next(t for t in src_tensors if t["name"] == name)
    ot = next(t for t in out_tensors if t["name"] == name)
    ne0, ne1, _ = ot["dims"]
    # 1. ggml on the new file
    row_new = L.ggml_row_size(GGML_TYPE_MXFP4, ne0)
    with open(a.out, "rb") as f:
        f.seek(ot["abs"] + (a.expert * ne1) * row_new)
        blob = f.read(row_new * ne1)
    new = np.zeros(ne0 * ne1, dtype=np.float32)
    L.dequantize_row_mxfp4(blob, new.ctypes.data, ne0 * ne1)
    new = new.reshape(ne1, ne0)
    # 2. the checkpoint's formula on the shard bytes
    wp, wo, wl, _, _ = shards.locate(f"layers.{a.layer}.ffn.experts.{a.expert}.{proj}.weight")
    sp, so, sl, _, _ = shards.locate(f"layers.{a.layer}.ffn.experts.{a.expert}.{proj}.scale")
    with open(wp, "rb") as f:
        f.seek(wo)
        w = np.frombuffer(f.read(wl), dtype=np.uint8).reshape(ne1, ne0 // 2)
        f.seek(so)
        s = np.frombuffer(f.read(sl), dtype=np.uint8).reshape(ne1, ne0 // QK)
    codes = np.empty((ne1, ne0), dtype=np.uint8)
    codes[:, 0::2] = w & 0x0F
    codes[:, 1::2] = w >> 4
    ref = E2M1[codes] * np.ldexp(1.0, s.astype(np.int32) - 127).astype(np.float32).repeat(QK, axis=1)
    exact = np.array_equal(new, ref)
    print(f"ggml MXFP4 dequant == checkpoint formula: {exact} (max |diff| {np.abs(new - ref).max():.3g})")
    # 3. the ROCMFP2 approximation of the same expert from the source file
    row_old = L.ggml_row_size(st["type"], ne0)
    with open(a.src, "rb") as f:
        f.seek(st["abs"] + (a.expert * ne1) * row_old)
        blob2 = f.read(row_old * ne1)
    old = np.zeros(ne0 * ne1, dtype=np.float32)
    L.rocmfpx_dequantize_row_fp2(blob2, old.ctypes.data, ne0 * ne1)
    old = old.reshape(ne1, ne0)
    c = np.corrcoef(new.ravel(), old.ravel())[0, 1]
    cT = np.corrcoef(new.ravel(), old.T.ravel()[: new.size])[0, 1] if old.T.size == new.size else float("nan")
    print(f"correlation with the source's ROCMFP2 expert: {c:.4f} (transposed: {cT:.4f}); "
          f"rel. RMS of the 2.5-bit error: {np.sqrt(((old - new) ** 2).mean() / (new ** 2).mean()):.4f}")
    return 0 if exact and c > 0.9 else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "check"):
        p = sub.add_parser(name)
        p.add_argument("--src", required=True)
        p.add_argument("--shards", required=True)
        p.add_argument("--out", required=True)
        if name == "build":
            p.add_argument("--wait", action="store_true")
            p.add_argument("--workers", type=int, default=12)
        else:
            p.add_argument("--layer", type=int, default=0)
            p.add_argument("--expert", type=int, default=0)
            p.add_argument("--which", default="gate", choices=("gate", "up", "down"))
    a = ap.parse_args()
    sys.exit(build(a) if a.cmd == "build" else check(a))


if __name__ == "__main__":
    main()
