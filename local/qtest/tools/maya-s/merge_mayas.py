#!/usr/bin/env python3
"""merge_mayas.py - one Lucebox-style GGUF for the strata-ds41 engine out of two sources:
  * the production file (MXFP4-ABL): the metadata (deepseek41.config etc.) and every non-expert tensor (Q8_0 attention /
    shared experts / head with the abliteration already applied, the F16 embedding, ...), copied byte for byte;
  * peasantsmith's Maya-S split (files 1-4 of 6): the routed-expert tensors blk.N.ffn_{gate,up,down}_exps.weight
    (IQ2_XXS/IQ3_XXS; IQ2_S/IQ3_S on layers 0-3 and 36-39), copied byte for byte with their types.
The engine reads neither llama.cpp-style metadata keys nor a Q8_0 embedding, so the split is not loadable as it is.
Tensor order = the production file's (the engine's layer-major reads keep their pattern).

  merge_mayas.py build --src OUR.gguf --shards DIR --out NEW.gguf
  merge_mayas.py check --src OUR.gguf --shards DIR --out NEW.gguf [--per-layer 6] [--seed 1]
    check: the written header equals the plan, every non-expert tensor equals the source, and per-layer random
    experts of each role equal the shard bytes.
"""
import argparse
import importlib.util
import os
import random
import sys
import time

CONV = "/home/yuya/ds41/convert_ds41_mxfp4.py"   # GGUF header read/write + ggml row sizes (the MXFP4 build tool)
_spec = importlib.util.spec_from_file_location("conv", CONV)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

SHARD = "DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS-0000%d-of-00006.gguf"
EXPERTS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
QUANT_NOTE = ("Maya-S routed experts (peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF: IQ2_XXS gate/up + IQ3_XXS down; "
              "IQ2_S gate/up + IQ3_S down on layers 0-3 and 36-39); Q8_0 attention/shared/head with the abliteration, "
              "F16 embedding, from the MXFP4-ABL file")
CHUNK = 64 << 20


def shard_tensors(d):
    m = {}
    for i in (1, 2, 3, 4):
        p = os.path.join(d, SHARD % i)
        _, _, ts, _, _ = conv.read_header(p)
        for t in ts:
            m[t["name"]] = (p, t)
    return m


def plan(a):
    L = conv.ggml()
    ver, kvs, src_t, align, _ = conv.read_header(a.src)
    sh = shard_tensors(a.shards)
    out, off = [], 0
    for t in src_t:
        is_exp = any(t["name"].endswith(e) for e in EXPERTS)
        if is_exp:
            p, st = sh[t["name"]]
            if list(st["dims"]) != list(t["dims"]):
                raise SystemExit("dims differ for %s: %s vs %s" % (t["name"], st["dims"], t["dims"]))
            nt = {"name": t["name"], "dims": list(t["dims"]), "type": st["type"], "src_path": p, "src_abs": st["abs"]}
        else:
            nt = {"name": t["name"], "dims": list(t["dims"]), "type": t["type"], "src_path": a.src, "src_abs": t["abs"]}
        nt["bytes"] = conv.tensor_bytes(L, nt)
        off = (off + align - 1) // align * align
        nt["offset"] = off
        nt["expert"] = is_exp
        off += nt["bytes"]
        out.append(nt)
    new_kvs = []
    for k, ty, v in kvs:
        if k == "general.name":
            v = "DeepSeek V4.1 Flash 1017 (Maya-S experts, MXFP4-ABL dense)"
        if k == "deepseek41.quantization":
            v = QUANT_NOTE
        new_kvs.append((k, ty, v))
    new_kvs.append(("general.source.experts", 8,
                    "peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF Maya-S files 1-4, HF revision d1fc0100 (2026-10-22)"))
    return ver, new_kvs, out, align, off


class Fds:
    def __init__(self):
        self.m = {}

    def __call__(self, p):
        if p not in self.m:
            self.m[p] = os.open(p, os.O_RDONLY)
        return self.m[p]

    def close(self):
        for x in self.m.values():
            os.close(x)


def copy(src, dst, soff, doff, n):
    done = 0
    while done < n:
        b = os.pread(src, min(CHUNK, n - done), soff + done)
        if not b:
            raise RuntimeError("short read: %d bytes at %d" % (n, soff))
        if os.pwrite(dst, b, doff + done) != len(b):
            raise RuntimeError("short write at %d" % (doff + done))
        os.posix_fadvise(src, soff + done, len(b), os.POSIX_FADV_DONTNEED)   # keep the page cache for production
        done += len(b)
    os.fdatasync(dst)
    os.posix_fadvise(dst, doff, n, os.POSIX_FADV_DONTNEED)


def same(fa, oa, fb, ob, n):
    done = 0
    while done < n:
        k = min(CHUNK, n - done)
        x = os.pread(fa, k, oa + done)
        y = os.pread(fb, k, ob + done)
        if x != y or len(x) != k:
            return False
        done += k
    return True


def build(a):
    ver, kvs, ts, align, total = plan(a)
    if os.path.exists(a.out):
        raise SystemExit(a.out + " exists (remove it first)")
    fds = Fds()
    with open(a.out, "wb") as f:
        data_start = conv.write_header(f, ver, kvs, ts, align)
    out_fd = os.open(a.out, os.O_WRONLY)
    os.ftruncate(out_fd, data_start + total)
    n_exp = sum(t["expert"] for t in ts)
    print("plan: %d tensors (%d expert), data %.2f GB at %d, file %.2f GB" % (len(ts), n_exp, total / 1e9, data_start,
          (data_start + total) / 1e9), flush=True)
    t0, done = time.time(), 0
    for i, t in enumerate(ts):
        copy(fds(t["src_path"]), out_fd, t["src_abs"], data_start + t["offset"], t["bytes"])
        done += t["bytes"]
        if t["expert"] or i % 100 == 0 or i + 1 == len(ts):
            el = max(time.time() - t0, 1e-9)
            print("%4d/%d %-32s type %3d %9.1f MB  total %7.1f GB  %.2f GB/s" % (i + 1, len(ts), t["name"], t["type"],
                  t["bytes"] / 1e6, done / 1e9, done / 1e9 / el), flush=True)
    os.fsync(out_fd)
    os.close(out_fd)
    fds.close()
    print("done: %s %d bytes in %.0f s" % (a.out, data_start + total, time.time() - t0), flush=True)


def check(a):
    ver, kvs, ts, align, total = plan(a)
    hv, hk, ht, halign, hds = conv.read_header(a.out)
    want = [(t["name"], t["dims"], t["type"], t["offset"]) for t in ts]
    got = [(t["name"], t["dims"], t["type"], t["offset"]) for t in ht]
    if want != got:
        print("check: the written header differs from the plan")
        return False
    size = os.path.getsize(a.out)
    if size != hds + total:
        print("check: file size %d, expected %d" % (size, hds + total))
        return False
    kv = {k: v for k, _, v in hk}
    print("check: header OK (%d tensors, align %d); quantization = %s" % (len(ht), halign, str(kv.get("deepseek41.quantization"))[:80]))
    rng = random.Random(a.seed)
    fds = Fds()
    out_fd = os.open(a.out, os.O_RDONLY)
    n = fails = 0
    types = {}
    for t in ts:
        s = fds(t["src_path"])
        if t["expert"]:
            ne = t["dims"][2]
            eb = t["bytes"] // ne
            for e in rng.sample(range(ne), a.per_layer):
                n += 1
                if not same(out_fd, hds + t["offset"] + e * eb, s, t["src_abs"] + e * eb, eb):
                    fails += 1
                    print("DIFFERS: %s expert %d" % (t["name"], e))
            types.setdefault(t["type"], 0)
            types[t["type"]] += 1
        else:
            n += 1
            if not same(out_fd, hds + t["offset"], s, t["src_abs"], t["bytes"]):
                fails += 1
                print("DIFFERS: %s" % t["name"])
    os.close(out_fd)
    fds.close()
    print("check: %d comparisons, %d differ; expert tensor types %s" % (n, fails, types))
    return fails == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "check"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-layer", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "build":
        build(a)
    else:
        sys.exit(0 if check(a) else 1)


if __name__ == "__main__":
    main()
