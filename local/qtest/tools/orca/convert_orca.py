#!/usr/bin/env python3
"""convert_orca.py - orcarouter's Mix-256G (a llama.cpp PR-30817 GGUF split) as one Lucebox-style GGUF for the strata-ds41
engine, keeping THEIR weights everywhere:
  * header/metadata: the production file's (deepseek41.config etc.; the engine reads Lucebox keys);
  * routed experts (120 tensors): their bytes and types (IQ2_S / IQ3_XXS / IQ3_S / MXFP4 per layer, gate = up);
  * every other tensor: their values in the type the engine's tensor check requires (= production's type):
      same type -> bytes copied; Q5_K / Q6_K -> Q8_0 (ggml dequantize_row_*, then ggml_quantize_chunk Q8_0);
      Q5_K / Q6_K / Q8_0 / F32 / BF16 -> F16 (ggml dequantize or bf16 -> f32, then ggml_fp32_to_fp16_row);
  * engram_q_norm / engram_k_norm: production's - their single engram_gate is bit-identical to q_norm * k_norm (asserted
    at build), and the engine only ever uses that product.
Their Engram table (shard 2) is not used: the engine's --engram keeps the Lucebox FP8 tables.

  convert_orca.py build --src OUR.gguf --shards 'DIR/NAME-0000?-of-00007.gguf' --out NEW.gguf
  convert_orca.py check (same arguments) [--per-layer 4]
"""
import argparse
import ctypes
import glob
import importlib.util
import os
import random
import sys
import time

import numpy as np

CONV = "/home/yuya/ds41/convert_ds41_mxfp4.py"
_spec = importlib.util.spec_from_file_location("conv", CONV)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

F32, F16, Q8_0, Q5_K, Q6_K, BF16 = 0, 1, 8, 13, 14, 30
EXPERTS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
OURS_ONLY = ("engram_q_norm.weight", "engram_k_norm.weight")
ROWS = 4096
LIB = ctypes.CDLL(conv.LIBGGML)
for fn in ("dequantize_row_q8_0", "dequantize_row_q5_K", "dequantize_row_q6_K"):
    getattr(LIB, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
LIB.ggml_bf16_to_fp32_row.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
LIB.ggml_fp32_to_fp16_row.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
LIB.ggml_quantize_chunk.restype = ctypes.c_size_t
LIB.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_void_p]
LIB.ggml_row_size.restype = ctypes.c_size_t
LIB.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
DEQ = {Q8_0: LIB.dequantize_row_q8_0, Q5_K: LIB.dequantize_row_q5_K, Q6_K: LIB.dequantize_row_q6_K}


def to_f32(ty, raw, n):
    if ty == F32:
        return np.frombuffer(raw, np.float32).copy()
    if ty == F16:
        return np.frombuffer(raw, np.float16).astype(np.float32)
    out = np.empty(n, np.float32)
    src = np.frombuffer(raw, np.uint8)
    if ty == BF16:
        LIB.ggml_bf16_to_fp32_row(src.ctypes.data, out.ctypes.data, n)
    else:
        DEQ[ty](src.ctypes.data, out.ctypes.data, n)
    return out


def from_f32(ty, x, rows, ne0):
    x = np.ascontiguousarray(x, np.float32)
    if ty == F32:
        return x.tobytes()
    if ty == F16:
        out = np.empty(x.size, np.uint16)
        LIB.ggml_fp32_to_fp16_row(x.ctypes.data, out.ctypes.data, x.size)
        return out.tobytes()
    if ty == Q8_0:
        out = np.empty(rows * LIB.ggml_row_size(Q8_0, ne0), np.uint8)
        n = LIB.ggml_quantize_chunk(Q8_0, x.ctypes.data, out.ctypes.data, 0, rows, ne0, None)
        assert n == out.size, (n, out.size)
        return out.tobytes()
    raise ValueError("no conversion to type %d" % ty)


def theirs_map(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit("no shards match " + pattern)
    m = {}
    for p in paths:
        _, _, ts, _, _ = conv.read_header(p)
        for t in ts:
            m[t["name"]] = (p, t)
    return m


def shard_path(out, s, n):
    if n == 1:
        return out
    base = out[:-5] if out.endswith(".gguf") else out
    return "%s-%05d-of-%05d.gguf" % (base, s + 1, n)


def plan(a):
    """-> (version, shards, tensors, align). A shard = {path, kvs, tensors, data}; tensor offsets are relative to ITS shard."""
    Lg = conv.ggml()
    ver, kvs, src_t, align, _ = conv.read_header(a.src)
    th = theirs_map(a.shards)
    ts = []
    for t in src_t:
        nt = {"name": t["name"], "dims": list(t["dims"])}
        if any(t["name"].endswith(e) for e in EXPERTS):
            p, s = th[t["name"]]
            nt.update(type=s["type"], kind="expert", src_path=p, src_abs=s["abs"], src_type=s["type"])
        elif t["name"].endswith(OURS_ONLY):
            nt.update(type=t["type"], kind="ours", src_path=a.src, src_abs=t["abs"], src_type=t["type"])
        else:
            if t["name"] not in th:
                raise SystemExit("their file has no " + t["name"])
            p, s = th[t["name"]]
            if list(s["dims"]) != list(t["dims"]):
                raise SystemExit("dims differ for %s: %s vs %s" % (t["name"], s["dims"], t["dims"]))
            kind = "copy" if s["type"] == t["type"] else "convert"
            nt.update(type=t["type"], kind=kind, src_path=p, src_abs=s["abs"], src_type=s["type"])
        nt["bytes"] = conv.tensor_bytes(Lg, nt)
        ts.append(nt)
    n = a.parts
    total_bytes = sum(t["bytes"] for t in ts)
    cum = 0
    for t in ts:                                   # consecutive tensors per shard, about equal bytes
        t["shard"] = min(n - 1, int(cum * n / total_bytes))
        cum += t["bytes"]
    offs = [0] * n
    for t in ts:
        k = t["shard"]
        offs[k] = (offs[k] + align - 1) // align * align
        t["offset"] = offs[k]
        offs[k] += t["bytes"]
    new_kvs = []
    for k, ty, v in kvs:
        if k == "general.name":
            v = "DeepSeek V4.1 Flash 1017 abliterated, orcarouter Mix-256G (their weights in the Lucebox layout)"
        if k == "deepseek41.quantization":
            v = ("orcarouter Mix-256G: their routed experts (IQ2_S / IQ3_XXS / IQ3_S / MXFP4 per layer) and their dense values "
                 "re-expressed in the engine's types (Q5_K/Q6_K -> Q8_0, small tensors -> F16)")
        new_kvs.append((k, ty, v))
    new_kvs.append(("general.source.weights", 8, "orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF Mix-256G shards 1,3-7, HF rev 43fac799"))
    align_kv = next(kv for kv in new_kvs if kv[0] == "general.alignment")
    shards = []
    for k in range(n):
        if n == 1:
            kk = list(new_kvs)
        else:
            split = [("split.no", 2, k), ("split.count", 2, n), ("split.tensors.count", 5, len(ts))]
            kk = list(new_kvs) + split if k == 0 else [align_kv] + split
        shards.append({"path": shard_path(a.out, k, n), "kvs": kk, "tensors": [t for t in ts if t["shard"] == k], "data": offs[k]})
    return ver, shards, ts, align


def rows_of(t):
    ne0 = t["dims"][0]
    rows = 1
    for d in t["dims"][1:]:
        rows *= d
    return ne0, rows


def produce(t, fd, r0, r1):
    """bytes of rows [r0, r1) of tensor t in its target type"""
    ne0, rows = rows_of(t)
    if t["kind"] in ("expert", "copy", "ours"):
        rb = t["bytes"] // rows
        return os.pread(fd, (r1 - r0) * rb, t["src_abs"] + r0 * rb)
    srb = LIB.ggml_row_size(t["src_type"], ne0)
    raw = os.pread(fd, (r1 - r0) * srb, t["src_abs"] + r0 * srb)
    return from_f32(t["type"], to_f32(t["src_type"], raw, (r1 - r0) * ne0), r1 - r0, ne0)


def engram_gate_check(a):
    th = theirs_map(a.shards)
    _, _, src_t, _, _ = conv.read_header(a.src)
    ours = {t["name"]: t for t in src_t}
    fd = os.open(a.src, os.O_RDONLY)
    for l in (1, 14):
        p, g = th["blk.%d.engram_gate.weight" % l]
        gfd = os.open(p, os.O_RDONLY)
        gv = np.frombuffer(os.pread(gfd, 5120 * 4 * 4, g["abs"]), np.float32)
        os.close(gfd)
        q = np.frombuffer(os.pread(fd, 5120 * 4 * 4, ours["blk.%d.engram_q_norm.weight" % l]["abs"]), np.float32)
        k = np.frombuffer(os.pread(fd, 5120 * 4 * 4, ours["blk.%d.engram_k_norm.weight" % l]["abs"]), np.float32)
        if not np.array_equal(gv, (q * k).astype(np.float32)):
            raise SystemExit("engram_gate of layer %d is not q_norm * k_norm" % l)
    os.close(fd)
    print("engram_gate == q_norm * k_norm on layers 1 and 14 (bit-identical)", flush=True)


def build(a):
    engram_gate_check(a)
    ver, shards, ts, align = plan(a)
    for sh in shards:
        if os.path.exists(sh["path"]):
            raise SystemExit(sh["path"] + " exists (remove it first)")
    kinds = {}
    for t in ts:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print("plan: %d tensors %s in %d shard(s): %s" % (len(ts), kinds, len(shards), ["%.1f GB" % (sh["data"] / 1e9) for sh in shards]), flush=True)
    src_fds = {}
    t0, done, i = time.time(), 0, 0
    for si, sh in enumerate(shards):
        with open(sh["path"], "wb") as f:
            data_start = conv.write_header(f, ver, sh["kvs"], sh["tensors"], align)
        out_fd = os.open(sh["path"], os.O_WRONLY)
        pos = 0    # strictly sequential appends: no pre-sizing (the NTFS driver fails a single huge extension)
        for t in sh["tensors"]:
            i += 1
            if t["offset"] > pos:
                os.pwrite(out_fd, b"\0" * (t["offset"] - pos), data_start + pos)
                pos = t["offset"]
            if t["src_path"] not in src_fds:
                src_fds[t["src_path"]] = os.open(t["src_path"], os.O_RDONLY)
            fd = src_fds[t["src_path"]]
            ne0, rows = rows_of(t)
            step = max(1, ROWS if t["kind"] == "convert" else (64 << 20) // max(1, t["bytes"] // rows))
            at = data_start + t["offset"]
            for r0 in range(0, rows, step):
                b = produce(t, fd, r0, min(rows, r0 + step))
                if os.pwrite(out_fd, b, at) != len(b):
                    raise RuntimeError("short write in " + t["name"])
                at += len(b)
            if at != data_start + t["offset"] + t["bytes"]:
                raise RuntimeError("%s: wrote %d bytes, planned %d" % (t["name"], at - data_start - t["offset"], t["bytes"]))
            pos = t["offset"] + t["bytes"]
            os.fdatasync(out_fd)   # keep production's page cache: sync and drop what was just read / written
            os.posix_fadvise(out_fd, data_start + t["offset"], t["bytes"], os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(fd, t["src_abs"], t["bytes"], os.POSIX_FADV_DONTNEED)
            done += t["bytes"]
            if t["kind"] == "expert" or i % 100 == 0 or i == len(ts):
                el = max(time.time() - t0, 1e-9)
                print("%4d/%d s%d %-34s %-7s %3d->%3d %8.1f MB  total %6.1f GB  %.2f GB/s" % (
                    i, len(ts), si + 1, t["name"], t["kind"], t["src_type"], t["type"], t["bytes"] / 1e6, done / 1e9, done / 1e9 / el), flush=True)
        os.fsync(out_fd)
        os.close(out_fd)
        print("shard %d/%d written: %s (%d bytes)" % (si + 1, len(shards), sh["path"], data_start + pos), flush=True)
    for fd in src_fds.values():
        os.close(fd)
    print("done: %d shard(s), %.1f GB of tensor data in %.0f s" % (len(shards), done / 1e9, time.time() - t0), flush=True)


def check(a):
    ver, shards, ts, align = plan(a)
    rng = random.Random(a.seed)
    src_fds = {}
    fails = n = 0
    rel = {}
    for si, sh in enumerate(shards):
        hv, hk, ht, halign, hds = conv.read_header(sh["path"])
        want = [(t["name"], t["dims"], t["type"], t["offset"]) for t in sh["tensors"]]
        got = [(t["name"], t["dims"], t["type"], t["offset"]) for t in ht]
        if want != got:
            print("check: shard %d header differs from the plan" % (si + 1))
            return False
        if os.path.getsize(sh["path"]) != hds + sh["data"]:
            print("check: shard %d size %d, expected %d" % (si + 1, os.path.getsize(sh["path"]), hds + sh["data"]))
            return False
        kv = {k: v for k, _, v in hk}
        if len(shards) > 1 and (kv.get("split.no") != si or kv.get("split.count") != len(shards) or kv.get("split.tensors.count") != len(ts)):
            print("check: shard %d split keys %s" % (si + 1, {k: kv.get(k) for k in ("split.no", "split.count", "split.tensors.count")}))
            return False
        print("check: shard %d/%d header OK (%d tensors, align %d)" % (si + 1, len(shards), len(ht), halign), flush=True)
        out_fd = os.open(sh["path"], os.O_RDONLY)
        for t in sh["tensors"]:
            if t["src_path"] not in src_fds:
                src_fds[t["src_path"]] = os.open(t["src_path"], os.O_RDONLY)
            fd = src_fds[t["src_path"]]
            ne0, rows = rows_of(t)
            rb = t["bytes"] // rows
            base = hds + t["offset"]
            if t["kind"] == "expert":
                eb = t["bytes"] // t["dims"][2]
                for e in rng.sample(range(t["dims"][2]), a.per_layer):
                    n += 1
                    if os.pread(out_fd, eb, base + e * eb) != os.pread(fd, eb, t["src_abs"] + e * eb):
                        fails += 1
                        print("DIFFERS: %s expert %d" % (t["name"], e))
            elif t["kind"] in ("copy", "ours"):
                n += 1
                if os.pread(out_fd, t["bytes"], base) != os.pread(fd, t["bytes"], t["src_abs"]):
                    fails += 1
                    print("DIFFERS: %s" % t["name"])
            else:
                r0 = rng.randrange(0, max(1, rows - 64))
                r1 = min(rows, r0 + 64)
                n += 1
                mine = os.pread(out_fd, (r1 - r0) * rb, base + r0 * rb)
                if mine != produce(t, fd, r0, r1):
                    fails += 1
                    print("DIFFERS (re-conversion): %s rows %d-%d" % (t["name"], r0, r1))
                srb = LIB.ggml_row_size(t["src_type"], ne0)
                x = to_f32(t["src_type"], os.pread(fd, (r1 - r0) * srb, t["src_abs"] + r0 * srb), (r1 - r0) * ne0)
                y = to_f32(t["type"], mine, (r1 - r0) * ne0)
                key = (t["name"].split(".", 2)[-1] if t["name"].startswith("blk.") else t["name"], t["src_type"], t["type"])
                rel.setdefault(key, []).append(float(np.sqrt(np.mean((y - x) ** 2)) / max(1e-30, np.sqrt(np.mean(x ** 2)))))
            os.posix_fadvise(out_fd, base, t["bytes"], os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(fd, t["src_abs"], t["bytes"], os.POSIX_FADV_DONTNEED)
        os.close(out_fd)
    for fd in src_fds.values():
        os.close(fd)
    print("check: %d comparisons, %d differ" % (n, fails))
    print("re-rounding of converted tensors (relative RMS of ours vs their values, sampled rows):")
    for k in sorted(rel):
        print("  %-36s %3d->%3d  mean %.2e  max %.2e" % (k[0], k[1], k[2], np.mean(rel[k]), np.max(rel[k])))
    return fails == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "check", "plan"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-layer", type=int, default=4)
    ap.add_argument("--parts", type=int, default=3, help="output shards (<name>-0000K-of-0000N.gguf); 1 = one file")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "plan":
        ver, shards, ts, align = plan(a)
        kinds = {}
        for t in ts:
            key = (t["kind"], t["src_type"], t["type"])
            kinds[key] = kinds.get(key, 0) + 1
        print("plan: %d tensors, expert bytes %.2f GB; shards: %s" % (len(ts), sum(t["bytes"] for t in ts if t["kind"] == "expert") / 1e9,
              ["%s %.1f GB %d tensors" % (os.path.basename(sh["path"]), sh["data"] / 1e9, len(sh["tensors"])) for sh in shards]))
        for k in sorted(kinds):
            print("  %-8s %3d -> %3d  x%d" % (k[0], k[1], k[2], kinds[k]))
    elif a.mode == "build":
        build(a)
    else:
        sys.exit(0 if check(a) else 1)


if __name__ == "__main__":
    main()
