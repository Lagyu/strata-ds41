#!/usr/bin/env python3
"""twin_noabl.py - switch the production file's TWIN (its second copy on the internal SSD; production reads only expert
blobs from it, via STRATA_BLOB_TWIN) between our abliterated dense tensors and DeepSeek's original ones, so that the twin
can serve as the exact "MXFP4 without ABL" model for a reference run. Production's own file on the P41 is never written.
  status       : how many of the 80 attn_output_b / ffn_down_shexp tensors hold the original / the abliterated bytes
  to-original  : write the original Q8_0 bytes from abliteration-backup.bin
  to-abl       : write the abliterated bytes back, read from the production file (checked against the recorded hash)
Every tensor is checked against the sha256 recorded when the abliteration was applied (old_sha256 = original, new_sha256 =
abliterated); 18 of the 80 are identical either way (strength 0). Idempotent.
"""
import hashlib
import importlib.util
import json
import os
import sys

TWIN = "/home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf"
PROD = "/mnt/win/llm/DeepSeek-V4.1-Flash-1017-MXFP4-ABL.gguf"
BK = "/home/yuya/models/DeepSeek-V4.1-Flash-1017-Lucebox/abliteration-backup.bin"
CONV = "/home/yuya/ds41/convert_ds41_mxfp4.py"
_spec = importlib.util.spec_from_file_location("conv", CONV)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    recs = json.load(open(BK + ".json"))["tensors"]
    tw = {t["name"]: t for t in conv.read_header(TWIN)[2]}
    pr = {t["name"]: t for t in conv.read_header(PROD)[2]}
    fd = os.open(TWIN, os.O_RDWR if mode in ("to-original", "to-abl") else os.O_RDONLY)
    bfd = os.open(BK, os.O_RDONLY)
    pfd = os.open(PROD, os.O_RDONLY)
    orig = abl = both = other = wrote = 0
    at = 0
    for r in recs:
        t = tw[r["name"]]
        n = r["bytes"]
        cur = sha(os.pread(fd, n, t["abs"]))
        same = r["old_sha256"] == r["new_sha256"]
        if same:
            both += cur == r["old_sha256"]
            other += cur != r["old_sha256"]
        elif mode == "status":
            orig += cur == r["old_sha256"]
            abl += cur == r["new_sha256"]
            other += cur not in (r["old_sha256"], r["new_sha256"])
        elif mode == "to-original":
            if cur == r["new_sha256"]:
                b = os.pread(bfd, n, at)
                assert sha(b) == r["old_sha256"], "backup bytes of %s do not match the recorded hash" % r["name"]
                os.pwrite(fd, b, t["abs"])
                wrote += 1
            elif cur != r["old_sha256"]:
                raise SystemExit("%s holds unexpected bytes - stopped" % r["name"])
        elif mode == "to-abl":
            if cur == r["old_sha256"]:
                b = os.pread(pfd, n, pr[r["name"]]["abs"])
                assert sha(b) == r["new_sha256"], "production's bytes of %s do not match the recorded hash" % r["name"]
                os.pwrite(fd, b, t["abs"])
                wrote += 1
            elif cur != r["new_sha256"]:
                raise SystemExit("%s holds unexpected bytes - stopped" % r["name"])
        at += n
    if mode in ("to-original", "to-abl"):
        os.fsync(fd)
        os.close(fd)
        fd = os.open(TWIN, os.O_RDONLY)
        want = "old_sha256" if mode == "to-original" else "new_sha256"
        bad = [r["name"] for r in recs if sha(os.pread(fd, r["bytes"], tw[r["name"]]["abs"])) != r[want]]
        print("%s: wrote %d tensors; verified %d/%d at the %s hashes%s" % (mode, wrote, len(recs) - len(bad), len(recs),
              "original" if mode == "to-original" else "abliterated", "" if not bad else " - MISMATCH " + str(bad[:3])))
        sys.exit(1 if bad else 0)
    print("twin status: %d original, %d abliterated, %d identical either way, %d other (of %d)" % (orig, abl, both, other, len(recs)))
    sys.exit(0 if other == 0 else 1)


if __name__ == "__main__":
    main()
