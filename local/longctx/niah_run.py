#!/usr/bin/env python3
"""niah_run.py - needle-in-a-haystack against the DS41 chat server, for the prefill scaling curve and a sanity check.

  niah_run.py --tokens 32768,131072,262144 [--depths 0.1,0.5,0.9] [--url ...] [--label L]

Haystack: public-domain books (/home/yuya/datasets/gutenberg/*.txt) concatenated, cut to the target token count with the
model's tokenizer; the needle (a specific sentence with a random 6-digit code) is inserted at the given depth; the
question asks for the code.  Records prefill tok/s (from the server's usage / wall time), the answer and pass/fail to
results.jsonl.  Lengths past the service's --ctx fail at the server ("exceed the context").
"""
import argparse, glob, json, os, random, re, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.environ.get("DS41_TOKENIZER", "/home/yuya/datasets/ds41-tokenizer/tokenizer.json")
BOOKS = os.environ.get("NIAH_BOOKS", "/home/yuya/datasets/gutenberg")
RESULTS = os.path.join(HERE, "results.jsonl")


def haystack_text():
    parts = []
    for f in sorted(glob.glob(os.path.join(BOOKS, "*.txt"))):
        s = open(f, errors="replace").read()
        # strip the Gutenberg header / footer
        a = s.find("*** START"); b = s.find("*** END")
        if a > 0: s = s[s.find("\n", a) + 1:]
        if b > 0: s = s[:b]
        parts.append(s)
    return "\n\n".join(parts)


def build(tok, text_ids, n_tokens, depth, code):
    body = text_ids[: n_tokens]
    needle = f"\n\nThe secret access code for the archive vault is {code}. Remember it.\n\n"
    needle_ids = tok.encode(needle, add_special_tokens=False).ids
    at = int(len(body) * depth)
    ids = body[:at] + needle_ids + body[at:]
    return tok.decode(ids)


def chat(url, messages, max_new):
    body = json.dumps({"model": "deepseek-v4.1-flash-1017", "messages": messages, "temperature": 0.0,
                       "max_tokens": max_new, "stream": False, "reasoning_effort": "none"}).encode()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("STRATA_API_KEY"):  # the server wants its key once lan.env is set
        headers["Authorization"] = "Bearer " + os.environ["STRATA_API_KEY"]
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=None) as resp:
        out = json.loads(resp.read().decode())
    return out, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="32768")
    p.add_argument("--depths", default="0.5")
    p.add_argument("--url", default="http://127.0.0.1:8150")
    p.add_argument("--label", default="niah")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    text = haystack_text()
    ids = tok.encode(text, add_special_tokens=False).ids
    print(f"haystack: {len(text)} chars, {len(ids)} tokens", flush=True)
    rng = random.Random(a.seed)
    for n in [int(x) for x in a.tokens.split(",")]:
        if n > len(ids):
            print(f"haystack too short for {n} tokens ({len(ids)})"); continue
        for d in [float(x) for x in a.depths.split(",")]:
            code = f"{rng.randint(100000, 999999)}"
            doc = build(tok, ids, n - 64, d, code)
            msgs = [{"role": "user", "content": doc + "\n\nQuestion: What is the secret access code for the archive vault? Answer with the six digits only."}]
            print(f"== {n} tokens, depth {d}, code {code}", flush=True)
            try:
                out, wall = chat(a.url, msgs, 32)
            except Exception as e:  # noqa: BLE001
                print("   request failed:", e, flush=True)
                open(RESULTS, "a").write(json.dumps({"label": a.label, "task": "niah", "tokens": n, "depth": d, "error": str(e)}) + "\n")
                continue
            text_out = out["choices"][0]["message"]["content"]
            usage = out.get("usage", {})
            ok = code in re.sub(r"\D", "", text_out)
            pt = usage.get("prompt_tokens") or n
            rec = {"label": a.label, "task": "niah", "tokens": n, "depth": d, "prompt_tokens": pt, "wall_s": round(wall, 1),
                   "prefill_tok_s_wall": round(pt / wall, 1), "timings": (out.get("usage") or {}).get("timings") or out.get("timings"), "pass": ok, "response": text_out[:100]}
            open(RESULTS, "a").write(json.dumps(rec) + "\n")
            print(f"   prompt_tokens {pt}  wall {wall:.0f} s  ({pt / wall:.1f} tok/s incl. everything)  pass {ok}  -> {text_out[:60]!r}", flush=True)


if __name__ == "__main__":
    main()
