#!/usr/bin/env python3
"""local/secprof/profile.py - drives the running service (started with STRATA_ROUTE_COUNTS) through the security
corpora so the engine counts the experts this work routes to: profiling prompts first (prompt processing only,
max_tokens 1), then find-and-fix tasks answered with thinking on (the experts used while writing analyses and fixes).
Resumable: requests already listed in OUT/progress.txt are skipped.
    python3 local/secprof/profile.py BUILD_DIR [--max-requests 80] [--gen 6]
"""
import argparse, json, os, time, urllib.request

URL = "http://127.0.0.1:8150/v1/chat/completions"
MODEL = "deepseek-v4.1-flash-1017"


def post(body, timeout=7200):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("--max-requests", type=int, default=80)
    ap.add_argument("--gen", type=int, default=6)
    ap.add_argument("--gen-tokens", type=int, default=2000)
    a = ap.parse_args()
    prog = os.path.join(a.build, "progress.txt")
    done = set(open(prog).read().split()) if os.path.exists(prog) else set()
    t_all, n_tok = time.time(), 0
    for i, line in enumerate(open(os.path.join(a.build, "prof_requests.jsonl"))):
        if i >= a.max_requests:
            break
        if "p%d" % i in done:
            continue
        r = json.loads(line)
        t0 = time.time()
        out = post({"model": MODEL, "messages": r["messages"], "max_tokens": 1, "temperature": 0, "reasoning_effort": "none"})
        n = out.get("usage", {}).get("prompt_tokens", 0)
        n_tok += n
        print("p%d %d tokens %.1f s (%.0f tok/s; %d tokens in %.0f s)" % (i, n, time.time() - t0, n / max(time.time() - t0, 1e-3),
                                                                        n_tok, time.time() - t_all), flush=True)
        open(prog, "a").write("p%d\n" % i)
    with open(os.path.join(a.build, "gen_answers.jsonl"), "a") as fa:
        for i, line in enumerate(open(os.path.join(a.build, "gen_tasks.jsonl"))):
            if i >= a.gen:
                break
            if "g%d" % i in done:
                continue
            r = json.loads(line)
            t0 = time.time()
            out = post({"model": MODEL, "messages": r["messages"], "max_tokens": a.gen_tokens, "temperature": 0})
            m = out["choices"][0]["message"]
            u = out.get("usage", {})
            fa.write(json.dumps({"id": r["id"], "reasoning": m.get("reasoning_content"), "content": m.get("content"),
                                 "usage": u}) + "\n")
            fa.flush()
            print("g%d %s: %s completion tokens in %.0f s" % (i, r["id"], u.get("completion_tokens"), time.time() - t0), flush=True)
            open(prog, "a").write("g%d\n" % i)
    print("profile done in %.0f s" % (time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
