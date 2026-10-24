#!/usr/bin/env python3
"""DeepSeek-V4.1 on luce_server: prefill/decode speed and greedy-output parity.

Runs a fixed prompt set at temperature 0 against /v1/chat/completions and
appends one JSON line per request to results.jsonl (label, timings, accept
rate, output hash and text). Two runs with different engine settings can be
compared with --compare LABEL_A LABEL_B (same prompt -> same greedy output?).

  python3 bench.py --label 3gpu-a            # run the set
  python3 bench.py --compare 2gpu-ref 3gpu-a # parity report
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time
import os
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"
DOC = pathlib.Path("/home/yuya/llm-src/lucebox/server/docs/DS4.md")


def long_prompt(n_chars):
    text = DOC.read_text(errors="replace")
    while len(text) < n_chars:
        text += "\n" + text
    return ("Read the following engine documentation and then answer: which "
            "environment variables control prefill, and what do they do?\n\n"
            + text[:n_chars])


PROMPTS = [
    ("code", "Write a Python function that returns the n-th Fibonacci number "
             "using fast doubling, with a short docstring and two doctests.", 256),
    ("prose", "Explain in three paragraphs why the sky is blue and why sunsets "
              "are red, for a curious twelve-year-old.", 256),
    ("ja", "日本の四季について、それぞれの季節の特徴を簡潔に説明してください。", 256),
    ("4k", None, 128),
]


# Calibration traffic for the expert usage profile (routing-stats.csv):
# a mix of what the box is used for.
CALIB = [
    "Write a detailed technical explanation of how a transformer language model processes a prompt, "
    "from tokenization to the final softmax, including attention and mixture-of-experts layers.",
    "PythonでLRUキャッシュをスレッドセーフに実装してください。設計の理由とテストコードも書いてください。",
    "Solve step by step: a train leaves at 9:40 at 72 km/h, a second one leaves the same station at 10:05 "
    "at 96 km/h on the same track. When and where does the second catch the first? Then generalize.",
    "東京から京都への二泊三日の旅行プランを、予算別に三つ提案してください。移動手段と見どころも含めて。",
    "Implement a small HTTP/1.1 server in C with epoll that serves static files, then review it for bugs.",
    "Summarize the causes and consequences of the 2008 financial crisis, then argue both for and against "
    "the bank bailouts.",
    "日本の少子高齢化について、原因、影響、政策の選択肢を論じてください。",
    "Write a Rust function that parses an INI file into a nested map, with error handling and unit tests.",
    "Prove that there are infinitely many primes, then explain the idea of the prime number theorem.",
    "Translate into natural Japanese and explain the nuances: 'The proof of the pudding is in the eating.' "
    "'Let sleeping dogs lie.' 'Every cloud has a silver lining.'",
    "Design a database schema for a library system with loans, reservations and fines; give SQL DDL and "
    "three example queries.",
    "Write a short story (about 500 words) about a lighthouse keeper who receives a letter from the future.",
]


CALIB2 = [
    "Explain the difference between TCP and QUIC, including congestion control and head-of-line blocking.",
    "以下のビジネスメールを丁寧な敬語に書き直してください：「明日の会議、資料まだ？早く送って。」",
    "Write a SQL query that finds the top 3 customers by revenue per region, using window functions, and explain it.",
    "A bag has 5 red and 7 blue balls. Two are drawn without replacement. What is the probability both are the "
    "same color? Show the reasoning step by step.",
    "Write a bash script that rotates log files older than 7 days, compresses them and keeps the last 30.",
    "量子コンピュータが従来のコンピュータと比べて得意な問題と苦手な問題を説明してください。",
    "Summarize the plot of Hamlet in 200 words and then list its main themes.",
    "Implement binary search in TypeScript with generics and explain the off-by-one pitfalls.",
    "What are the pros and cons of microservices versus a modular monolith for a team of 10 engineers?",
    "江戸時代の参勤交代制度の目的と、それが経済に与えた影響について説明してください。",
    "Derive the formula for the sum of a geometric series and give two applications.",
    "Write a Go program that reads a CSV of transactions and prints per-account balances, with error handling.",
    "Give me a weekly vegetarian meal plan with a shopping list, aiming for 120 g of protein per day.",
    "機械学習における過学習とは何か、防ぐための具体的な手法を5つ挙げて説明してください。",
    "Explain how public-key cryptography works using RSA, with a tiny numeric example.",
    "Write a haiku sequence (five haiku) about autumn in Kyoto, then translate them into English.",
]


def calibrate(args):
    prompts = CALIB2 if args.calib_set == 2 else CALIB
    for i, prompt in enumerate(prompts):
        r = request(args.url, args.model, prompt, args.calib_tokens, args.timeout)
        print(f"calib {i:2d}: prompt {r['prompt_tokens']} tok, out {r['completion_tokens']} tok, "
              f"decode {r['decode_tok_s'] or 0:.1f} tok/s", flush=True)


REASONING_EFFORT = None   # --reasoning-effort: sent as "reasoning_effort" (Strata's server: none = thinking off)


def request(url, model, prompt, max_tokens, timeout):
    req_body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if REASONING_EFFORT:
        req_body["reasoning_effort"] = REASONING_EFFORT
    body = json.dumps(req_body).encode()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("STRATA_API_KEY"):  # the server wants its key once lan.env is set (strata-ds41)
        headers["Authorization"] = "Bearer " + os.environ["STRATA_API_KEY"]
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body, headers=headers)
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.load(r)
    wall = time.monotonic() - t0
    choice = res["choices"][0]
    msg = choice["message"]
    text = (msg.get("reasoning_content") or "") + "\n<<answer>>\n" + (msg.get("content") or "")
    usage = res.get("usage", {})
    t = usage.get("timings") or res.get("timings") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    prefill_ms = t.get("prefill_ms", 0.0)
    # Lucebox reports prefill_ms / decode_tokens_per_sec; Strata's server uses llama.cpp's names
    # (prompt_ms, prompt_per_second, predicted_per_second) and counts its drafts.
    if not prefill_ms and t.get("prompt_ms"):
        prefill_ms = t["prompt_ms"]
    prefill_tok_s = t.get("prompt_per_second") or (round(prompt_tokens / (prefill_ms / 1000.0), 1) if prefill_ms else None)
    accept = usage.get("accept_rate")
    if accept is None and t.get("drafts_offered"):
        accept = round((t.get("drafts_accepted") or 0) / t["drafts_offered"], 2)
    if accept is None and t.get("draft_n"):   # Strata's DS41 engine: llama.cpp's draft_n / draft_n_accepted
        accept = round((t.get("draft_n_accepted") or 0) / t["draft_n"], 2)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "prefill_ms": prefill_ms,
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": t.get("decode_tokens_per_sec") or t.get("predicted_per_second"),
        "accept_rate": accept,
        "spec_decode_ran": usage.get("spec_decode_ran"),
        "cache_hit": t.get("cache_hit"),
        "wall_s": round(wall, 2),
        "finish_reason": choice.get("finish_reason"),
        "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "text": text,
    }


def run(args):
    with RESULTS.open("a") as out:
        for rep in range(args.repeat):
            for name, prompt, max_tokens in PROMPTS:
                if args.only and name not in args.only.split(","):
                    continue
                if prompt is None:
                    # A run-unique first line, so the prefix cache (also on
                    # disk, across restarts) cannot serve the long prompt.
                    prompt = f"[bench run {args.label} {time.time():.0f} rep {rep}]\n" + long_prompt(args.long_chars)
                r = request(args.url, args.model, prompt, max_tokens, args.timeout)
                r.update({"label": args.label, "prompt": name, "rep": rep,
                          "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{args.label:14s} {name:6s} rep{rep} prompt {r['prompt_tokens']:5d} tok "
                      f"prefill {r['prefill_tok_s'] or 0:7.1f} tok/s  decode {r['decode_tok_s'] or 0:5.1f} tok/s "
                      f"accept {r['accept_rate'] or 0:.2f}  out {r['completion_tokens']} tok  {r['sha256']}",
                      flush=True)


def compare(a, b):
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        if r["label"] in (a, b) and r.get("rep", 0) == 0:
            by.setdefault(r["prompt"], {})[r["label"]] = r
    same = 0
    for name, d in by.items():
        if a not in d or b not in d:
            continue
        ta, tb = d[a]["text"], d[b]["text"]
        n = 0
        while n < min(len(ta), len(tb)) and ta[n] == tb[n]:
            n += 1
        ok = ta == tb
        same += ok
        print(f"{name:6s} {'identical' if ok else f'diverge at char {n} of {len(ta)}/{len(tb)}'}")
        if not ok:
            print(f"   {a}: ...{ta[max(0, n - 40):n + 60]!r}")
            print(f"   {b}: ...{tb[max(0, n - 40):n + 60]!r}")
    print(f"{same} identical")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="deepseek-v4.1-flash-1017")
    ap.add_argument("--label", default="run")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--long-chars", type=int, default=15000)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib-tokens", type=int, default=768)
    ap.add_argument("--calib-set", type=int, default=1)
    args = ap.parse_args()
    global REASONING_EFFORT
    REASONING_EFFORT = args.reasoning_effort
    if args.compare:
        compare(*args.compare)
    elif args.calibrate:
        calibrate(args)
    else:
        run(args)


if __name__ == "__main__":
    sys.exit(main())
