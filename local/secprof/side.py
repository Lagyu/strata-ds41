#!/usr/bin/env python3
"""local/secprof/side.py LABEL [--tasks N] - the held-out find-and-fix tasks (side_tasks.jsonl) answered by the running
service, greedy, thinking on; answers to BUILD/side-LABEL.jsonl for a side-by-side read (production vs a splice)."""
import argparse, json, os, time, urllib.request

B = "/home/yuya/datasets/security-profile/build"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2500)
    a = ap.parse_args()
    out = os.path.join(B, "side-%s.jsonl" % a.label)
    have = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    with open(out, "a") as f:
        for i, line in enumerate(open(os.path.join(B, "side_tasks.jsonl"))):
            if i >= a.tasks:
                break
            r = json.loads(line)
            if r["id"] in have:
                continue
            body = {"model": "deepseek-v4.1-flash-1017", "messages": r["messages"], "max_tokens": a.max_tokens, "temperature": 0}
            req = urllib.request.Request("http://127.0.0.1:8150/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            m = json.loads(urllib.request.urlopen(req, timeout=7200).read())
            msg, u = m["choices"][0]["message"], m.get("usage", {})
            f.write(json.dumps({"id": r["id"], "reasoning": msg.get("reasoning_content"), "content": msg.get("content"),
                                "usage": u, "seconds": time.time() - t0}) + "\n")
            f.flush()
            print("%s %s: %s tokens in %.0f s" % (a.label, r["id"], u.get("completion_tokens"), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
