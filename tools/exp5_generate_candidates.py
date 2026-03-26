#!/usr/bin/env python3
"""
exp5_generate_candidates.py — Generator with chain-aware prompting + variable temperature

Two prompt modes:
  flat:        Uses {evidence} placeholder (same as exp1b/exp4)
  chain_aware: Uses {hop1_title}, {hop1_text}, {hop2_title}, {hop2_text}, {question}
               from the top-ranked chain. Forces two-hop reasoning.

Also supports --fallback_temperature for increasing diversity on the seed+i
fallback path (where the 7B model tends to repeat itself).

Usage:
    python3 tools/exp5_generate_candidates.py \
        --evidence      exp1b/evidence/dev_K100_chains.jsonl \
        --gold          data/hotpotqa/raw/hotpot_dev_distractor_v1.json \
        --prompt_file   exp5/inputs/prompt_v4_chain_aware_7b.txt \
        --prompt_mode   chain_aware \
        --out_jsonl     exp5/candidates/dev_M5_7b_chain_aware.jsonl \
        --manifest      exp5/manifest.json \
        --llm_base_url  http://127.0.0.1:8000/v1 \
        --llm_model_id  /var/tmp/u24sf51014/sro/models/qwen2.5-7b-instruct \
        --split dev --m 5 --seed 12345 \
        --temperature 0.7 \
        --fallback_temperature 1.0
"""
import argparse, hashlib, json, os, time, re
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib import request, error

SCHEMA_VERSION = "sro-proof.exp5.candidates.v1"
FINAL_RE = re.compile(r"<final>\s*(.*?)\s*(?:</final>|\Z)", re.DOTALL | re.IGNORECASE)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def read_text(path):
    with open(path, "r", encoding="utf-8") as f: return f.read()

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: yield json.loads(line)

def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_done_qids(path):
    done = set()
    if not os.path.exists(path): return done
    for line in open(path):
        line = line.strip()
        if line:
            try: done.add(str(json.loads(line)["qid"]))
            except: pass
    return done

def update_manifest(path, patch):
    obj = {}
    if os.path.exists(path):
        with open(path, "r") as f: obj = json.load(f)
    obj.update(patch)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)


# ── evidence formatting ──

def evidence_text_flat(chains, max_chars):
    parts = []
    for ch in chains:
        cid = ch.get("chain_id", "?")
        for hop in ch.get("hops", []):
            parts.append(f"[chain {cid} hop {hop.get('hop','?')}] "
                        f"{hop.get('title','')}: {hop.get('text','')}")
        parts.append("")
    s = "\n".join(parts).strip()
    if len(s) > max_chars: s = s[:max_chars].rsplit("\n", 1)[0]
    return s

def get_top_chain_hops(chains):
    """Extract (hop1_title, hop1_text, hop2_title, hop2_text) from top chain."""
    if not chains:
        return "", "", "", ""
    top = chains[0]
    hops = top.get("hops", [])
    h1_title = h1_text = h2_title = h2_text = ""
    if len(hops) >= 1:
        h1_title = hops[0].get("title", "").strip()
        h1_text = hops[0].get("text", "").strip()
    if len(hops) >= 2:
        h2_title = hops[1].get("title", "").strip()
        h2_text = hops[1].get("text", "").strip()
    return h1_title, h1_text, h2_title, h2_text


# ── answer extraction ──

def extract_final(raw):
    if not raw: return ""
    if "</think>" in raw: raw = raw.split("</think>")[-1].strip()
    matches = FINAL_RE.findall(raw)
    if matches:
        ans = (matches[-1] or "").strip()
        lines = [l.strip() for l in ans.splitlines() if l.strip()]
        return " ".join(lines).strip() if lines else ""
    for line in raw.splitlines():
        line = line.strip()
        if line: return line
    return ""

def is_bad(ans):
    a = ans.strip()
    if not a: return True
    low = a.lower()
    if low.startswith("[chain"): return True
    if "if the evidence does not contain" in low: return True
    if low.startswith("the evidence provided"): return True
    if low.startswith(("okay,", "alright,", "so,")): return True
    if low in {"unknown", "unk"}: return True
    if len(a) > 120: return True
    return False


# ── LLM client ──

class ExpectedN5Failure(RuntimeError): pass

def call_completions(base_url, model, prompt, temperature, top_p, max_tokens,
                     seed, timeout, n, stop):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "<final>"},
        ],
        "n": int(n), "temperature": float(temperature),
        "top_p": float(top_p), "max_tokens": int(max_tokens), "seed": int(seed),
    }
    if stop: payload["stop"] = stop
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body_text[:500]}") from e
    choices = body.get("choices", [])
    return [c["message"]["content"] for c in choices]

def retry_call(fn, retries=3, sleep_s=2.0):
    for attempt in range(retries + 1):
        try: return fn()
        except Exception as e:
            if attempt == retries: raise
            time.sleep(sleep_s * (attempt + 1))


# ── main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--prompt_mode", choices=["flat", "chain_aware"], required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--chains_for_llm", type=int, default=5)
    ap.add_argument("--max_evidence_chars", type=int, default=6000)
    ap.add_argument("--llm_base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--llm_model_id", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--fallback_temperature", type=float, default=None,
                    help="Temperature for seed+i fallback (default: same as --temperature)")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--timeout_s", type=int, default=60)
    ap.add_argument("--http_retries", type=int, default=3)
    ap.add_argument("--http_retry_sleep", type=float, default=2.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force_seed_loop", action="store_true")
    args = ap.parse_args()

    fallback_temp = args.fallback_temperature or args.temperature

    prompt_t = read_text(args.prompt_file)
    prompt_hash = hashlib.sha256(prompt_t.encode()).hexdigest()

    print(f"[exp5] Prompt mode: {args.prompt_mode}")
    print(f"[exp5] Prompt: {args.prompt_file} (sha256={prompt_hash[:16]}...)")
    print(f"[exp5] Model: {args.llm_model_id}")
    print(f"[exp5] M={args.m}  T={args.temperature}  fallback_T={fallback_temp}  "
          f"top_p={args.top_p}  seed={args.seed}")

    done = load_done_qids(args.out_jsonl) if args.resume else set()
    if done: print(f"[exp5] Resuming: {len(done)} already done")

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    STOP_TOKENS = ["</final>"]

    update_manifest(args.manifest, {
        "exp5_generate": {
            "schema_version": SCHEMA_VERSION,
            "prompt_mode": args.prompt_mode,
            "prompt_hash": prompt_hash,
            "model": args.llm_model_id,
            "temperature": args.temperature,
            "fallback_temperature": fallback_temp,
            "m": args.m,
            "seed": args.seed,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    })

    wrote = skipped = n5_ok = n5_fallback = 0

    for evd in iter_jsonl(args.evidence):
        qid = str(evd["qid"])
        if qid in done:
            skipped += 1
            continue

        q = evd["question"]
        all_chains = evd.get("chains", evd.get("evidence", {}).get("chains", []))
        fixed_chains = all_chains[:args.chains_for_llm]
        shown_chain_ids = [int(ch.get("chain_id", i)) for i, ch in enumerate(fixed_chains)]

        # ── Build prompt based on mode ──
        if args.prompt_mode == "chain_aware":
            h1_title, h1_text, h2_title, h2_text = get_top_chain_hops(all_chains)
            prompt = prompt_t.format(
                hop1_title=h1_title, hop1_text=h1_text,
                hop2_title=h2_title, hop2_text=h2_text,
                question=q
            )
        else:
            ev_txt = evidence_text_flat(fixed_chains, args.max_evidence_chars)
            prompt = prompt_t.format(question=q, evidence=ev_txt)

        sampling_strategy = "n=5"
        answers = []

        try:
            if args.force_seed_loop:
                raise ExpectedN5Failure("forced via --force_seed_loop")

            def _n5():
                return call_completions(
                    args.llm_base_url, args.llm_model_id, prompt,
                    args.temperature, args.top_p, args.max_new_tokens,
                    args.seed, args.timeout_s, n=args.m, stop=STOP_TOKENS)

            raws = retry_call(_n5, retries=args.http_retries,
                              sleep_s=args.http_retry_sleep)

            if len(raws) != args.m:
                raise ExpectedN5Failure(f"server returned {len(raws)}, expected {args.m}")

            answers = [extract_final(r) for r in raws]

            if len(set(answers)) == 1:
                raise ExpectedN5Failure("n=5 returned identical outputs")

            empty = sum(1 for a in answers if not a.strip())
            if empty >= (args.m // 2 + 1):
                raise ExpectedN5Failure(f"n=5 produced {empty}/{args.m} empty")

            n5_ok += 1

        except ExpectedN5Failure as e:
            n5_fallback += 1
            if n5_fallback <= 10 or n5_fallback % 500 == 0:
                print(f"[exp5] qid={qid} fallback seed+i: {e}")
            sampling_strategy = "seed+i"
            answers = []
            for i in range(args.m):
                def _n1(i=i):
                    return call_completions(
                        args.llm_base_url, args.llm_model_id, prompt,
                        fallback_temp, args.top_p, args.max_new_tokens,
                        args.seed + i, args.timeout_s, n=1, stop=STOP_TOKENS)
                raws = retry_call(_n1, retries=args.http_retries,
                                  sleep_s=args.http_retry_sleep)
                a = extract_final(raws[0]) if raws else ""
                answers.append(a)

        except Exception as e:
            raise RuntimeError(f"Unexpected error for qid={qid}: {e}") from e

        rec = {
            "schema_version": SCHEMA_VERSION,
            "split": args.split,
            "qid": qid,
            "candidates": [{"answer_id": i, "answer_text": answers[i]}
                          for i in range(args.m)],
            "generation_context": {
                "sampling_strategy": sampling_strategy,
                "chains_shown_to_llm": shown_chain_ids,
                "prompt_mode": args.prompt_mode,
                "prompt_hash": prompt_hash,
                "temperature": args.temperature if sampling_strategy == "n=5" else fallback_temp,
                "seed": args.seed,
            },
            "stats": {
                "bad_count": sum(1 for a in answers if is_bad(a)),
                "unique_count": len(set(answers)),
                "empty_count": sum(1 for a in answers if not a.strip()),
            },
        }

        append_jsonl(args.out_jsonl, rec)
        wrote += 1

        if wrote % 200 == 0:
            pct = (wrote + skipped) / 7405 * 100
            print(f"[exp5] {wrote} wrote  {skipped} skipped  "
                  f"n5_ok={n5_ok} fallback={n5_fallback}  ({pct:.1f}%)")

    print(f"\n[exp5] DONE: wrote={wrote}  skipped={skipped}  "
          f"n5_ok={n5_ok}  fallback={n5_fallback}")

    update_manifest(args.manifest, {
        "exp5_generate_done": {
            "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wrote": wrote, "skipped": skipped,
            "n5_ok": n5_ok, "fallback": n5_fallback,
        }
    })

if __name__ == "__main__":
    main()