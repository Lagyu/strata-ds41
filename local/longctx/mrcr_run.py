#!/usr/bin/env python3
"""mrcr_run.py - OpenAI MRCR (openai/mrcr, 2/4/8 needles, up to 1M tokens) against the DS41 chat server.

  mrcr_run.py select  [--min-tokens N --max-tokens N --needles 2,4,8 --per 1]   list candidate rows (DeepSeek tokens)
  mrcr_run.py run     --rows 2needle_1:17,8needle_0:42 [--url ...] [--max-new 1024] [--label L]

A row's prompt is a JSON list of chat messages; the model must answer the Nth occurrence of a hidden request, prefixed
by `random_string_to_prepend`.  Score = SequenceMatcher ratio after the prefix (0 without it), as the dataset's own
grader.  Token counts use the model's tokenizer (deepseek-ai/DeepSeek-V4.1-Flash-1017 tokenizer.json): the server
counts the same way, plus the chat template's few tokens per message.  Results append to results.jsonl.
"""
import argparse, glob, json, os, sys, time, urllib.request
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("MRCR_DIR", "/home/yuya/datasets/mrcr")
TOK = os.environ.get("DS41_TOKENIZER", "/home/yuya/datasets/ds41-tokenizer/tokenizer.json")
RESULTS = os.path.join(HERE, "results.jsonl")


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOK)


def grade(response, answer, prefix):
    if not response.startswith(prefix):
        return 0.0
    response = response.removeprefix(prefix)
    answer = answer.removeprefix(prefix)
    return float(SequenceMatcher(None, response, answer).ratio())


def rows_of(needles):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(os.path.join(DATA, f"{needles}needle", "*.parquet"))):
        t = pq.read_table(f)
        name = os.path.basename(f).removesuffix(".parquet")
        for i in range(t.num_rows):
            yield name, i, {c: t.column(c)[i].as_py() for c in t.column_names}


def count_tokens(tok, prompt_json):
    msgs = json.loads(prompt_json)
    n = 0
    for m in msgs:
        n += len(tok.encode(m["content"], add_special_tokens=False).ids) + 4   # ~4 template tokens a message
    return n, len(msgs)


def cmd_select(a):
    tok = tokenizer()
    picked = []
    for needles in [int(x) for x in a.needles.split(",")]:
        cands = []
        for name, i, r in rows_of(needles):
            # a cheap char pre-filter (4.25 chars/token on English prose), then the exact count
            est = r["n_chars"] / 4.25
            if est < a.min_tokens * 0.8 or est > a.max_tokens * 1.25:
                continue
            n, nm = count_tokens(tok, r["prompt"])
            if a.min_tokens <= n <= a.max_tokens:
                cands.append((n, name, i, nm, r["n_chars"], len(r["answer"])))
        cands.sort()
        # spread the picks over the window
        step = max(1, len(cands) // max(1, a.per))
        for n, name, i, nm, nc, na in cands[::step][: a.per]:
            print(f"{needles}needle {name}:{i}  tokens {n}  messages {nm}  chars {nc}  answer_chars {na}")
            picked.append(f"{name}:{i}")
    print("rows:", ",".join(picked))


def chat(url, messages, max_new, timeout=None):
    body = json.dumps({"model": "deepseek-v4.1-flash-1017", "messages": messages, "temperature": 0.0,
                       "max_tokens": max_new, "stream": False, "reasoning_effort": "none"}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    return out, time.time() - t0


def cmd_run(a):
    tok = tokenizer()
    order = []
    for item in a.rows.split(","):
        name, _, idx = item.partition(":")
        order.append((name, int(idx)))
    want = set(order)
    found = {}
    for needles in (2, 4, 8):
        if not any(n.startswith(f"{needles}needle") for n, _ in want):
            continue
        for name, i, r in rows_of(needles):
            if (name, i) in want:
                found[(name, i)] = (needles, r)
    for key in order:   # the --rows order (interleave the needle counts so every count gets a result early)
        if key not in found:
            print("row not found:", key); continue
        needles, r = found[key]
        name, i = key
        if True:
            msgs = json.loads(r["prompt"])
            n, nm = count_tokens(tok, r["prompt"])
            ans_tokens = len(tok.encode(r["answer"], add_special_tokens=False).ids)
            max_new = a.max_new or min(4096, ans_tokens + 64)
            label = a.label.replace("900k", "1m") if n > 950_000 else a.label   # the chain's stage name was written for ~900K rows
            print(f"== {name}:{i}  needles {needles}  ~{n} tokens  {nm} messages  answer {ans_tokens} tokens; max_new {max_new}", flush=True)
            try:
                out, wall = chat(a.url, msgs, max_new)
            except Exception as e:  # noqa: BLE001
                print("   request failed:", e, flush=True)
                rec = {"label": label, "row": f"{name}:{i}", "needles": needles, "tokens_est": n, "error": str(e)}
                open(RESULTS, "a").write(json.dumps(rec) + "\n")
                continue
            text = out["choices"][0]["message"]["content"]
            usage = out.get("usage", {})
            score = grade(text, r["answer"], r["random_string_to_prepend"])
            rec = {"label": label, "row": f"{name}:{i}", "needles": needles, "tokens_est": n,
                   "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                   "wall_s": round(wall, 1), "score": round(score, 4), "prefix_ok": text.startswith(r["random_string_to_prepend"]),
                   "timings": (out.get("usage") or {}).get("timings") or out.get("timings"), "response_head": text[:200], "answer_head": r["answer"][:200]}
            open(RESULTS, "a").write(json.dumps(rec) + "\n")
            print(f"   prompt_tokens {usage.get('prompt_tokens')}  wall {wall:.0f} s  score {score:.3f}  prefix_ok {rec['prefix_ok']}", flush=True)
            print("   response:", text[:160].replace("\n", " "), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--min-tokens", type=int, default=850_000)
    s.add_argument("--max-tokens", type=int, default=980_000)
    s.add_argument("--needles", default="2,4,8")
    s.add_argument("--per", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--rows", required=True)
    r.add_argument("--url", default="http://127.0.0.1:8150")
    r.add_argument("--max-new", type=int, default=0)
    r.add_argument("--label", default="mrcr")
    a = p.parse_args()
    {"select": cmd_select, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
