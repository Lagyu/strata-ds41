#!/usr/bin/env python3
"""splice_b.py - the residency-aware expert splice (local/quant-research-2026-10-21.md, option B).

The production file holds every routed expert in MXFP4 (17.93 MiB).  Decode time goes to the experts that are not
resident, and how many are resident depends on their size.  This tool keeps the most-used experts (the head of the
--rank order the engine also places by) in MXFP4, byte for byte, and requantizes the rest from those MXFP4 blocks to
smaller ggml types - by default the next --warm experts to IQ3_XXS and the remainder to IQ2_XXS - one expert at a
time, with that expert's columns of unsloth's imatrix.  The output is v41_gguf.py's mixed-tier layout, which
ds41_geometry.hpp / Ds41FileBlobs read: per layer and type a stacked tensor of the experts at that type, in expert
order (`blk.L.ffn_{gate,up}_exps.<type>.weight` [n_embd, n_ff, n_t], `blk.L.ffn_down_exps.<type>.weight`
[n_ff, n_embd, n_t]), plus the I32 maps `blk.L.ffn_exps_gu_type` / `blk.L.ffn_exps_down_type` [n_expert].  Every other
tensor and all metadata are copied unchanged.

    splice_b.py rank  [--csv routing-stats.csv ...] [--imatrix IMAT ...] [--weights 0.5,0.5] --out RANK.bin
    splice_b.py plan  --rank R.bin --hot 2741 --warm 7540 [--warm-type iq3_xxs] [--cold-type iq2_xxs]
                      [--imatrix IMAT] --out PLAN.json
    splice_b.py quant --plan PLAN.json --src SRC.gguf --imatrix IMAT --qdir QDIR [--layers A-B] [--threads N]
    splice_b.py write --plan PLAN.json --src SRC.gguf --qdir QDIR --out OUT.gguf
    splice_b.py check --plan PLAN.json --src SRC.gguf --qdir QDIR --out OUT.gguf [--samples N]

A type is a ggml type name or id; "gu,down" gives gate/up and down different types (e.g. --cold-type iq2_xxs,iq3_xxs).
rank merges routing counts (Lucebox's per-prompt profile, an imatrix's calibration counts) into one order; the engine
places experts by the same file (--rank), so the MXFP4 head is also what it keeps closest.
quant is resumable and incremental: QDIR/<type>/LNN.bin holds a layer's experts requantized to that type (each as its
gate rows, up rows, down rows), LNN.json their offsets and imatrix-weighted errors; only experts missing there are
quantized, so a second plan pays only for the experts it adds.
Needs ggml's base library built as a shared object (GGML_LIB, default local/ggml-shared/src/libggml-base.so; built
from third_party/llama.cpp/ggml with -DBUILD_SHARED_LIBS=ON - the engine's own ggml sources, so MXFP4 decodes the same).
"""
import argparse
import concurrent.futures as cf
import ctypes
import json
import os
import re
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MXFP4, I32 = 39, 26
TYPE_IDS = {"f32": 0, "f16": 1, "q8_0": 8, "q2_k": 10, "q3_k": 11, "q4_k": 12, "iq2_xxs": 16, "iq2_xs": 17,
            "iq3_xxs": 18, "iq3_s": 21, "iq2_s": 22, "iq4_xs": 23, "mxfp4": 39}
TYPE_NAMES = {v: k for k, v in TYPE_IDS.items()}
EXP_RE = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def parse_type(s):
    s = s.strip().lower()
    return int(s) if s.isdigit() else TYPE_IDS[s]


def parse_pair(s):
    p = [parse_type(x) for x in s.split(",")]
    return (p[0], p[-1])


# ================================================================== ggml through ctypes
class _Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


class Ggml:
    _TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)

    def __init__(self):
        path = os.environ.get("GGML_LIB", os.path.join(ROOT, "local/ggml-shared/src/libggml-base.so"))
        self.lib = lib = ctypes.CDLL(path)
        lib.ggml_get_type_traits.restype = ctypes.POINTER(_Traits)
        lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        lib.ggml_row_size.restype = ctypes.c_size_t
        lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        lib.ggml_quantize_init.argtypes = [ctypes.c_int]
        lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
        self._to_float = {}

    def row_size(self, t, n):
        return int(self.lib.ggml_row_size(t, n))

    def nbytes(self, t, dims):
        n = 1
        for d in dims[1:]:
            n *= int(d)
        return self.row_size(t, int(dims[0])) * n

    def name(self, t):
        return self.lib.ggml_get_type_traits(t).contents.type_name.decode()

    def to_float(self, t, raw, n):
        f = self._to_float.get(t)
        if f is None:
            f = self._to_float[t] = self._TO_FLOAT(self.lib.ggml_get_type_traits(t).contents.to_float)
        out = np.empty(n, np.float32)
        src = np.frombuffer(raw, np.uint8)
        f(src.ctypes.data, out.ctypes.data, n)
        return out

    def quantize(self, t, x, nrows, n_per_row, imat):
        out = np.empty(self.row_size(t, n_per_row) * nrows, np.uint8)
        im = None if imat is None else np.ascontiguousarray(imat, np.float32)
        done = self.lib.ggml_quantize_chunk(t, x.ctypes.data, out.ctypes.data, 0, nrows, n_per_row,
                                            None if im is None else im.ctypes.data)
        assert done == out.nbytes, (done, out.nbytes)
        return out


# ================================================================== GGUF header (raw: the KV section is copied as bytes)
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _str(buf, p):
    n, = struct.unpack_from("<Q", buf, p)
    return bytes(buf[p + 8:p + 8 + n]).decode("utf-8", "replace"), p + 8 + n


def _val(buf, p, vt):
    if vt in _SCALAR:
        v, = struct.unpack_from(_SCALAR[vt], buf, p)
        return v, p + struct.calcsize(_SCALAR[vt])
    if vt == 8:
        return _str(buf, p)
    if vt == 9:
        et, n = struct.unpack_from("<IQ", buf, p)
        p += 12
        if et in _SCALAR:   # numbers: skip in one step, keep them only when short
            w = struct.calcsize(_SCALAR[et])
            v = list(struct.unpack_from("<%d%s" % (n, _SCALAR[et][1]), buf, p)) if n <= 4096 else None
            return v, p + w * n
        v = []
        for _ in range(n):
            x, p = _val(buf, p, et)
            v.append(x)
        return v, p
    raise ValueError("GGUF value type %d" % vt)


class Gguf:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(1 << 20)
            magic, self.version, nt, nkv = struct.unpack_from("<4sIQQ", head, 0)
            assert magic == b"GGUF", path
            size = 1 << 20
            while True:   # grow the window until the whole header parses
                try:
                    self._parse(head, nt, nkv)
                    break
                except struct.error:
                    size *= 4
                    f.seek(0)
                    head = f.read(size)
        self.by_name = {t[0]: t for t in self.tensors}

    def _parse(self, buf, nt, nkv):
        p = 24
        self.kv = {}
        self.align = 32
        for _ in range(nkv):
            k, p = _str(buf, p)
            vt, = struct.unpack_from("<I", buf, p)
            v, p = _val(buf, p + 4, vt)
            self.kv[k] = v
        self.kv_bytes = bytes(buf[24:p])
        self.kv_count = nkv
        self.align = int(self.kv.get("general.alignment", 32))
        self.tensors = []
        for _ in range(nt):
            name, p = _str(buf, p)
            nd, = struct.unpack_from("<I", buf, p)
            dims = struct.unpack_from("<%dQ" % nd, buf, p + 4)
            ty, off = struct.unpack_from("<IQ", buf, p + 4 + 8 * nd)
            p += 4 + 8 * nd + 12
            self.tensors.append((name, tuple(int(d) for d in dims), int(ty), int(off)))
        self.data_start = (p + self.align - 1) // self.align * self.align

    def offset(self, name):
        return self.data_start + self.by_name[name][3]


# ================================================================== plan
def load_rank(path):
    a = np.fromfile(path, np.int32).reshape(-1, 2)
    return [(int(l), int(e)) for l, e in a]


def imatrix_counts(path):
    """{layer: counts[n_expert]} from an imatrix GGUF (the routed-token counts its calibration saw per expert)."""
    g = Gguf(path)
    out = {}
    with open(path, "rb") as f:
        for name, dims, ty, off in g.tensors:
            m = re.match(r"blk\.(\d+)\.ffn_gate_exps\.weight\.counts$", name)
            if m and ty == 0:
                n = int(np.prod(dims))
                f.seek(g.data_start + off)
                out[int(m.group(1))] = np.frombuffer(f.read(4 * n), np.float32).copy()
    return out


def lucebox_counts(path):
    """[layers, experts] routed-token counts from Lucebox's routing-stats.csv ('#' comment lines first)."""
    return np.stack([np.array([int(x) for x in ln.split(",")], np.float64)
                     for ln in open(path) if ln.strip() and not ln.startswith("#")])


def cmd_rank(a):
    """A --rank file (hottest (layer, expert) first, int32 pairs) from several routing-count sources, each normalized
    to its share of the routes so a big corpus does not outvote a small one: score = sum_i w_i * share_i."""
    srcs = []
    for p in a.csv or []:
        srcs.append((p, lucebox_counts(p)))
    for p in a.imatrix or []:
        c = imatrix_counts(p)
        srcs.append((p, np.stack([c[l] for l in sorted(c)]).astype(np.float64)))
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(srcs)
    assert len(w) == len(srcs), "one weight per source"
    score = sum(wi * c / c.sum() for wi, (_, c) in zip(w, srcs))
    L, E = score.shape
    order = np.argsort(-score.reshape(-1), kind="stable")
    np.stack([order // E, order % E], axis=1).astype(np.int32).tofile(a.out)
    for p, c in srcs:
        cum = np.cumsum(c.reshape(-1)[order]) / c.sum()
        log("%s: the new order's top 2741 / 5000 / 10281 cover %.1f / %.1f / %.1f %% of its routes" % (
            os.path.basename(p), 100 * cum[2740], 100 * cum[4999], 100 * cum[10280]))
    log("%d x %d -> %s" % (L, E, a.out))


def cmd_plan(a):
    rank = load_rank(a.rank)
    hot, warm, cold = (MXFP4, MXFP4), parse_pair(a.warm_type), parse_pair(a.cold_type)
    n_layers = max(l for l, _ in rank) + 1
    n_exp = max(e for _, e in rank) + 1
    gu = [[-1] * n_exp for _ in range(n_layers)]
    dn = [[-1] * n_exp for _ in range(n_layers)]
    cls = {}
    for p, (l, e) in enumerate(rank):
        c = "hot" if p < a.hot else "warm" if p < a.hot + a.warm else "cold"
        gu[l][e], dn[l][e] = {"hot": hot, "warm": warm, "cold": cold}[c]
        cls[(l, e)] = c
    assert all(t >= 0 for row in gu for t in row), "the rank file does not list every expert"
    D, FF = a.n_embd, a.n_ff
    g = Ggml()
    size = {c: 2 * g.nbytes(t[0], (D, FF)) + g.nbytes(t[1], (FF, D)) for c, t in (("hot", hot), ("warm", warm), ("cold", cold))}
    count = {c: sum(1 for v in cls.values() if v == c) for c in ("hot", "warm", "cold")}
    summary = {c: {"experts": count[c], "gu_type": t[0], "d_type": t[1], "mib_each": size[c] / 2**20,
                   "gib": count[c] * size[c] / 2**30}
               for c, t in (("hot", hot), ("warm", warm), ("cold", cold))}
    if a.imatrix:   # what share of the calibration's routed tokens each class serves
        cnt = imatrix_counts(a.imatrix)
        tot = sum(float(v.sum()) for v in cnt.values())
        for c in summary:
            summary[c]["route_share"] = sum(float(cnt[l][e]) for (l, e), v in cls.items() if v == c and l in cnt) / tot
    plan = {"rank": os.path.abspath(a.rank), "hot": a.hot, "warm": a.warm, "n_embd": D, "n_ff": FF,
            "summary": summary, "layers": {str(l): {"gu_type": gu[l], "d_type": dn[l]} for l in range(n_layers)}}
    json.dump(plan, open(a.out, "w"))
    tot_gib = sum(s["gib"] for s in summary.values())
    for c, s in summary.items():
        log("%-4s %5d experts  %-7s/%-7s %6.2f MiB each  %6.1f GiB%s" % (
            c, s["experts"], TYPE_NAMES.get(s["gu_type"], s["gu_type"]), TYPE_NAMES.get(s["d_type"], s["d_type"]),
            s["mib_each"], s["gib"], "  %4.1f %% of routed tokens" % (100 * s["route_share"]) if "route_share" in s else ""))
    log("routed experts %.1f GiB (MXFP4 everywhere: %.1f GiB) -> %s" % (tot_gib, len(rank) * size["hot"] / 2**30, a.out))


# ================================================================== quant
class Imatrix:
    """Per-expert importance (mean squared activation per input column) for gate/up and down of each layer."""

    def __init__(self, path):
        self.g = Gguf(path)
        self.mm = np.memmap(path, np.uint8, "r")

    def _t(self, name):
        _, dims, ty, off = self.g.by_name[name]
        assert ty == 0, name
        n = int(np.prod(dims))
        o = self.g.data_start + off
        return np.frombuffer(self.mm[o:o + 4 * n], np.float32).reshape(tuple(reversed(dims)))

    def layer(self, L, role):
        s = self._t("blk.%d.ffn_%s_exps.weight.in_sum2" % (L, role))      # [n_expert, n_in]
        c = self._t("blk.%d.ffn_%s_exps.weight.counts" % (L, role))[:, 0]  # [n_expert]
        mean = s.sum(0) / max(float(c.sum()), 1.0)                        # unseen experts: the layer's average
        w = np.where(c[:, None] > 0, s / np.maximum(c, 1.0)[:, None], mean[None, :]).astype(np.float32)
        return w, c


def _werr(g, t, q, x, nrows, n, w):
    """imatrix-weighted relative squared error of the requantized rows."""
    y = g.to_float(t, q.tobytes(), nrows * n).reshape(nrows, n)
    x = x.reshape(nrows, n)
    d = ((y - x) ** 2) @ w
    s = (x ** 2) @ w
    return float(d.sum() / max(float(s.sum()), 1e-30))


class QStore:
    """QDIR/<class>/LNN.{bin,json}: one layer's requantized experts at one (gate/up, down) type pair, <class> = the
    type name, or "gu-down" when they differ.  Kept per type pair, not per plan, so plans that share a class share
    its experts (a variant quantizes only the experts it adds); a store only grows (appended, then the json)."""

    def __init__(self, g, qdir):
        self.g, self.qdir, self.meta, self.fd = g, qdir, {}, {}

    def paths(self, L, gt, dt):
        n = self.g.name(gt).lower()
        if dt != gt:
            n += "-" + self.g.name(dt).lower()
        d = os.path.join(self.qdir, n)
        return os.path.join(d, "L%02d.bin" % L), os.path.join(d, "L%02d.json" % L)

    def experts(self, L, gt, dt):
        k = (L, gt, dt)
        if k not in self.meta:
            jp = self.paths(L, gt, dt)[1]
            self.meta[k] = json.load(open(jp))["experts"] if os.path.exists(jp) else {}
        return self.meta[k]

    def where(self, L, gt, dt, e, role):
        """(fd, offset) of one requantized expert's role rows."""
        k = (L, gt, dt)
        if k not in self.fd:
            self.fd[k] = os.open(self.paths(L, gt, dt)[0], os.O_RDONLY)
        return self.fd[k], self.experts(L, gt, dt)[str(e)][role]


def cmd_quant(a):
    plan = json.load(open(a.plan))
    g = Ggml()
    src = Gguf(a.src)
    imat = Imatrix(a.imatrix)
    store = QStore(g, a.qdir)
    D, FF = plan["n_embd"], plan["n_ff"]
    fd = os.open(a.src, os.O_RDONLY)
    lo, hi = (int(x) for x in a.layers.split("-")) if a.layers else (0, len(plan["layers"]) - 1)
    for t in {x for lay in plan["layers"].values() for x in lay["gu_type"] + lay["d_type"]} - {MXFP4}:
        g.lib.ggml_quantize_init(t)
    t_all, n_all = time.time(), 0
    for L in range(lo, hi + 1):
        lay = plan["layers"][str(L)]
        gu_t, d_t = lay["gu_type"], lay["d_type"]
        work = {}
        for e in range(len(gu_t)):
            k = (gu_t[e], d_t[e])
            if k != (MXFP4, MXFP4) and str(e) not in store.experts(L, *k):
                work.setdefault(k, []).append(e)
        if not work:
            continue
        roles = {}
        for r in ("gate", "up", "down"):
            name, dims, ty, off = src.by_name["blk.%d.ffn_%s_exps.weight" % (L, r)]
            assert ty == MXFP4, "%s is %d, the splice reads MXFP4 experts" % (name, ty)
            roles[r] = (src.data_start + off, g.nbytes(ty, dims[:2]))
        w_gu, _ = imat.layer(L, "gate")
        w_up, _ = imat.layer(L, "up")
        w_dn, _ = imat.layer(L, "down")

        def one(e):
            out, err = [], {}
            for r, n_in, nrows, t, w in (("gate", D, FF, gu_t[e], w_gu[e]), ("up", D, FF, gu_t[e], w_up[e]),
                                         ("down", FF, D, d_t[e], w_dn[e])):
                base, per = roles[r]
                raw = os.pread(fd, per, base + e * per)
                if t == MXFP4:    # a mixed expert (one role kept): its MXFP4 bytes as they are
                    out.append(np.frombuffer(raw, np.uint8))
                    continue
                x = g.to_float(MXFP4, raw, nrows * n_in)
                q = g.quantize(t, x, nrows, n_in, w)
                err[r] = _werr(g, t, q, x, nrows, n_in, w)
                out.append(q)
            return e, out, err

        for (gt, dt), todo in sorted(work.items()):
            t0 = time.time()
            bp, jp = store.paths(L, gt, dt)
            os.makedirs(os.path.dirname(bp), exist_ok=True)
            have = store.experts(L, gt, dt)
            errs = []
            with open(bp, "ab") as f, cf.ThreadPoolExecutor(a.threads) as ex:
                pos = os.fstat(f.fileno()).st_size   # bytes a run cut short appended stay unreferenced
                for e, out, err in ex.map(one, todo):   # in expert order
                    offs = []
                    for b in out:
                        offs.append(pos)
                        f.write(b.tobytes())
                        pos += b.nbytes
                    have[str(e)] = {"gate": offs[0], "up": offs[1], "down": offs[2], "err": err}
                    errs += list(err.values())
                f.flush()
                os.fsync(f.fileno())
            json.dump({"layer": L, "gu_type": gt, "d_type": dt, "experts": have}, open(jp + ".tmp", "w"))
            os.replace(jp + ".tmp", jp)
            n_all += len(todo)
            log("L%02d %s: %d experts in %.0f s, weighted rel. err median %.4f max %.4f (%d done this run, %.0f s)" % (
                L, os.path.basename(os.path.dirname(bp)), len(todo), time.time() - t0, float(np.median(errs)),
                max(errs), n_all, time.time() - t_all))
    log("quant done: %d experts in %.0f s" % (n_all, time.time() - t_all))


# ================================================================== write
def layout(plan, src, g):
    """The output tensor list: (name, dims, type, source) with source = ("copy", src_offset, nbytes) |
    ("map", int32 array) | ("stack", layer, role, type, [experts])."""
    D, FF = plan["n_embd"], plan["n_ff"]
    out = []
    for name, dims, ty, off in src.tensors:
        m = EXP_RE.match(name)
        if not m:
            out.append((name, dims, ty, ("copy", src.data_start + off, g.nbytes(ty, dims))))
            continue
        L, r = int(m.group(1)), m.group(2)
        lay = plan["layers"][str(L)]
        tv = lay["d_type"] if r == "down" else lay["gu_type"]
        kinds = sorted(set(tv))
        shape = (FF, D) if r == "down" else (D, FF)
        if len(kinds) == 1:   # uniform role: the usual stacked tensor at that type
            if kinds[0] == ty:
                out.append((name, dims, ty, ("copy", src.data_start + off, g.nbytes(ty, dims))))
            else:
                out.append((name, shape + (len(tv),), kinds[0], ("stack", L, r, kinds[0], list(range(len(tv))))))
            continue
        if r != "up":   # the maps: gate carries gu's, down its own
            out.append(("blk.%d.ffn_exps_%s_type" % (L, "down" if r == "down" else "gu"), (len(tv),), I32,
                        ("map", np.array(tv, np.int32))))
        for t in kinds:
            ex = [e for e in range(len(tv)) if tv[e] == t]
            out.append(("blk.%d.ffn_%s_exps.%s.weight" % (L, r, g.name(t).lower()), shape + (len(ex),), t,
                        ("stack", L, r, t, ex)))
    return out


def only_layers(tl, spec):
    """layout() restricted to the globals + blocks A-B (a partial test file)."""
    lo, hi = (int(x) for x in spec.split("-"))
    return [t for t in tl if not t[0].startswith("blk.") or lo <= int(t[0].split(".")[1]) <= hi]


def cmd_write(a):
    plan = json.load(open(a.plan))
    g = Ggml()
    src = Gguf(a.src)
    tl = layout(plan, src, g)
    al = src.align
    kv, n_kv = src.kv_bytes, src.kv_count
    if a.layers:   # a partial test file (test_ds41_moe): the globals + these blocks, as v41_gguf.py --layers
        keep = list(range(*(lambda lo, hi: (lo, hi + 1))(*(int(x) for x in a.layers.split("-")))))
        tl = only_layers(tl, a.layers)
        key = b"strata.v41.partial_layers"
        kv += struct.pack("<Q", len(key)) + key + struct.pack("<IIQ", 9, 4, len(keep)) + struct.pack("<%dI" % len(keep), *keep)
        n_kv += 1
    # header: the source's KV section byte for byte, then the new tensor infos
    info = bytearray()
    offs, pos = [], 0
    for name, dims, ty, s in tl:
        nb = s[2] if s[0] == "copy" else s[1].nbytes if s[0] == "map" else g.nbytes(ty, dims)
        nm = name.encode()
        info += struct.pack("<Q", len(nm)) + nm + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
        info += struct.pack("<IQ", ty, pos)
        offs.append((pos, nb))
        pos = (pos + nb + al - 1) // al * al
    head = struct.pack("<4sIQQ", b"GGUF", src.version, len(tl), n_kv) + kv + bytes(info)
    data_start = (len(head) + al - 1) // al * al
    total = data_start + pos
    log("%d tensors, data %.1f GiB -> %s" % (len(tl), pos / 2**30, a.out))
    store = QStore(g, a.qdir)
    sfd = os.open(a.src, os.O_RDONLY)
    tmp = a.out + ".tmp"
    ofd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.pwrite(ofd, head + b"\0" * (data_start - len(head)), 0)

    def copy(fd_in, off_in, n, off_out):
        while n > 0:
            try:
                k = os.copy_file_range(fd_in, ofd, min(n, 1 << 30), off_in, off_out)
            except OSError:   # another filesystem (EXDEV) or no in-kernel copy: through user space
                k = os.pwrite(ofd, os.pread(fd_in, min(n, 64 << 20), off_in), off_out)
            if k <= 0:
                raise OSError("short copy (%d) at %d" % (k, off_out))
            n -= k
            off_in += k
            off_out += k

    t0, done = time.time(), 0
    for (name, dims, ty, s), (o, nb) in zip(tl, offs):
        at = data_start + o
        if s[0] == "copy":
            copy(sfd, s[1], nb, at)
        elif s[0] == "map":
            os.pwrite(ofd, s[1].tobytes(), at)
        else:
            _, L, r, t, ex = s
            per = nb // len(ex)
            lay = plan["layers"][str(L)]
            for k, e in enumerate(ex):
                if t == MXFP4:
                    _, sd, sty, soff = src.by_name["blk.%d.ffn_%s_exps.weight" % (L, r)]
                    copy(sfd, src.data_start + soff + e * per, per, at + k * per)
                else:
                    qf, qo = store.where(L, lay["gu_type"][e], lay["d_type"][e], e, r)
                    copy(qf, qo, per, at + k * per)
        done += nb
        if s[0] != "copy" or nb > 1 << 30:
            el = time.time() - t0
            sys.stdout.write("\r%s %.1f / %.1f GiB (%.0f MB/s)   " % (time.strftime("%H:%M:%S"), done / 2**30,
                                                                     pos / 2**30, done / max(el, 1e-3) / 1e6))
            sys.stdout.flush()
    os.ftruncate(ofd, total)
    os.fsync(ofd)
    os.close(ofd)
    os.replace(tmp, a.out)
    print()
    log("wrote %s (%.1f GiB) in %.0f s" % (a.out, total / 2**30, time.time() - t0))


# ================================================================== check
def cmd_check(a):
    """Reads the output back: every tensor where layout() put it, a sample of experts byte-equal to their sources."""
    plan = json.load(open(a.plan))
    g = Ggml()
    src, out = Gguf(a.src), Gguf(a.out)
    tl = layout(plan, src, g)
    if a.layers:
        tl = only_layers(tl, a.layers)
    assert [t[0] for t in tl] == [t[0] for t in out.tensors], "tensor list differs"
    for (name, dims, ty, s), (n2, d2, t2, o2) in zip(tl, out.tensors):
        assert (tuple(dims), ty) == (d2, t2), name
    rng = np.random.default_rng(0)
    sfd, ofd = os.open(a.src, os.O_RDONLY), os.open(a.out, os.O_RDONLY)
    store = QStore(g, a.qdir)
    n_ok = 0
    stacks = [x for x in tl if x[3][0] == "stack"]
    for i in rng.choice(len(stacks), size=min(a.samples, len(stacks)), replace=False):
        name, dims, ty, (_, L, r, t, ex) = stacks[i]
        per = g.nbytes(ty, dims) // len(ex)
        k = int(rng.integers(len(ex)))
        e = ex[k]
        got = os.pread(ofd, per, out.offset(name) + k * per)
        if t == MXFP4:
            _, sd, sty, soff = src.by_name["blk.%d.ffn_%s_exps.weight" % (L, r)]
            want = os.pread(sfd, per, src.data_start + soff + e * per)
        else:
            lay = plan["layers"][str(L)]
            qf, qo = store.where(L, lay["gu_type"][e], lay["d_type"][e], e, r)
            want = os.pread(qf, per, qo)
        assert got == want, "%s expert %d differs" % (name, e)
        n_ok += 1
    for name, dims, ty, s in tl:
        if s[0] == "map":
            got = np.frombuffer(os.pread(ofd, s[1].nbytes, out.offset(name)), np.int32)
            assert (got == s[1]).all(), name
    log("check OK: %d tensors, %d sampled experts byte-equal to their sources, maps equal" % (len(tl), n_ok))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("rank")
    p.add_argument("--csv", action="append")
    p.add_argument("--imatrix", action="append")
    p.add_argument("--weights")
    p.add_argument("--out", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--rank", required=True)
    p.add_argument("--hot", type=int, required=True)
    p.add_argument("--warm", type=int, required=True)
    p.add_argument("--warm-type", default="iq3_xxs")
    p.add_argument("--cold-type", default="iq2_xxs")
    p.add_argument("--n-embd", type=int, default=5120)
    p.add_argument("--n-ff", type=int, default=2304)
    p.add_argument("--imatrix")
    p.add_argument("--out", required=True)
    p = sp.add_parser("quant")
    for k in ("--plan", "--src", "--imatrix", "--qdir"):
        p.add_argument(k, required=True)
    p.add_argument("--layers")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    for c in ("write", "check"):
        p = sp.add_parser(c)
        for k in ("--plan", "--src", "--qdir", "--out"):
            p.add_argument(k, required=True)
        p.add_argument("--samples", type=int, default=64)
        p.add_argument("--layers", help="write: a partial test file with only blocks A-B (+ the globals)")
    a = ap.parse_args()
    {"rank": cmd_rank, "plan": cmd_plan, "quant": cmd_quant, "write": cmd_write, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
