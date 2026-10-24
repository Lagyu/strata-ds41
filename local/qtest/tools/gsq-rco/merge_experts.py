#!/usr/bin/env python3
"""merge_experts.py - one Lucebox-style GGUF for the strata-ds41 engine out of two sources (generalized merge_mayas.py):
  * the production file (MXFP4-ABL): the metadata (deepseek41.config etc.) and every non-expert tensor (Q8_0 attention /
    shared experts / head with the abliteration already applied, the F16 embedding, ...), copied byte for byte;
  * a candidate quant's llama.cpp split (--shards GLOB): only the routed-expert tensors blk.N.ffn_{gate,up,down}_exps.weight,
    copied byte for byte with their own types.
The engine reads neither llama.cpp-style metadata keys nor other dense types/names, so a candidate split is not loadable as
it is; this also makes the abliteration exact (the dense bytes are production's). Tensor order = the production file's.

  merge_experts.py build --src OUR.gguf --shards 'DIR/NAME-*-of-0000N.gguf' --out NEW.gguf --name NAME --note NOTE --source SRC
  merge_experts.py check (same arguments) [--per-layer 6] [--seed 1]
    check: the written header equals the plan, every non-expert tensor equals production's, and per-layer random experts
    of each role equal the shard bytes.
"""
import argparse
import glob
import importlib.util
import os
import random
import sys
import time

CONV = "/home/yuya/ds41/convert_ds41_mxfp4.py"   # GGUF header read/write + ggml row sizes (the MXFP4 build tool)
_spec = importlib.util.spec_from_file_location("conv", CONV)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

EXPERTS = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
CHUNK = 64 << 20


def shard_tensors(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit("no shards match " + pattern)
    m = {}
    for p in paths:
        _, _, ts, _, _ = conv.read_header(p)
        for t in ts:
            if any(t["name"].endswith(e) for e in EXPERTS):
                m[t["name"]] = (p, t)
    return m


def plan(a):
    L = conv.ggml()
    ver, kvs, src_t, align, _ = conv.read_header(a.src)
    sh = shard_tensors(a.shards)
    out, off, used = [], 0, set()
    for t in src_t:
        is_exp = any(t["name"].endswith(e) for e in EXPERTS)
        if is_exp:
            if t["name"] not in sh:
                raise SystemExit("the shards have no " + t["name"])
            p, st = sh[t["name"]]
            used.add(t["name"])
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
    extra = set(sh) - used
    if extra:
        raise SystemExit("expert tensors in the shards that production lacks: %s" % sorted(extra)[:5])
    new_kvs = []
    for k, ty, v in kvs:
        if k == "general.name":
            v = a.name
        if k == "deepseek41.quantization":
            v = a.note
        new_kvs.append((k, ty, v))
    new_kvs.append(("general.source.experts", 8, a.source))
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
            types[t["type"]] = types.get(t["type"], 0) + 1
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
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--per-layer", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "build":
        build(a)
    else:
        sys.exit(0 if check(a) else 1)


if __name__ == "__main__":
    main()
