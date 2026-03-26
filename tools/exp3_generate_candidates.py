#!/usr/bin/env python3
"""
exp3_generate_candidates.py — Exp3: chain-aware prompting for candidate generation

Fork of exp1_generate_candidates.py (v4_qwen) with ONE change:
  The evidence_text function now labels hop structure explicitly when
  --evidence_format=chain_aware.

Everything else is IDENTICAL to exp1b generation:
  - Same model (Qwen2.5-1.5B-Instruct)
  - Same M=5 candidates
  - Same temperature/top_p/seed
  - Same prefill/stop strategy
  - Same n=5 → seed+i fallback

This lets us measure oracle@5 improvement from ONLY the prompt change.

Usage (from project root):
    /var/tmp/u24sf51014/sro/conda-envs/llm/bin/python3 tools/exp3_generate_candidates.py \\
        --evidence      exp1b/evidence/dev_K100_chains.jsonl \\
        --gold          data/hotpotqa/raw/hotpot_dev_distractor_v1.json \\
        --prompt_file   exp3/inputs/prompt_v3_chain_aware.txt \\
        --prompt_version v3_chain_aware \\
        --evidence_format chain_aware \\
        --out_jsonl     exp3/candidates/dev_M5_candidates_chain_aware.jsonl \\
        --manifest      exp3/manifest.json \\
        --llm_base_url  http://127.0.0.1:8000/v1 \\
        --llm_model_id  qwen2.5-1.5b-instruct \\
        --split dev --m 5 --seed 12345
"""
import argparse, hashlib, json, os, time, re
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib import request, error

SCHEMA_VERSION = "sro-proof.exp3.candidates.v1_chain_aware"

FINAL_RE = re.compile(r"<final>\s*(.*?)\s*(?:</final>|\Z)", re.DOTALL | re.IGNORECASE)

# ─────────────────────────── utils ──────────────────────────────────

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_done_qids(out_jsonl: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["qid"]))
            except Exception:
                continue
    return done

def update_manifest(path: str, patch: Dict[str, Any]) -> None:
    obj: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    obj.update(patch)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def write_sha256_file(path: str, sha_path: str) -> None:
    h = sha256_file(path)
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(h + "  " + os.path.basename(path) + "\n")


# ─────────────────────────── evidence formatting ────────────────────

def evidence_text_flat(chains: List[Dict[str, Any]], max_chars: int) -> str:
    """Original exp1b flat evidence format — identical to exp1_generate_candidates.py."""
    parts: List[str] = []
    for ch in chains:
        cid = ch.get("chain_id", "?")
        for hop in ch.get("hops", []):
            parts.append(
                f"[chain {cid} hop {hop.get('hop','?')}] "
                f"{hop.get('title','')}: {hop.get('text','')}"
            )
        parts.append("")
    s = "\n".join(parts).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rsplit("\n", 1)[0]
    return s


def evidence_text_chain_aware(chains: List[Dict[str, Any]], max_chars: int) -> str:
    """
    Chain-aware evidence format — the ONE change in Exp3.

    Instead of:
        [chain 0 hop 1] Title1: Text1
        [chain 0 hop 2] Title2: Text2

    Produces:
        === Chain 1 of 5 ===
        Bridge document: Title1
        Text1

        Answer document: Title2
        Text2

    This explicitly tells the LLM:
      - Each chain is a 2-hop reasoning path
      - Document 1 (bridge) connects the question to related info
      - Document 2 (answer) is where the final answer typically lives
      - Both documents are needed together
    """
    parts: List[str] = []
    n_chains = len(chains)
    for ci, ch in enumerate(chains):
        hops = ch.get("hops", [])
        parts.append(f"=== Chain {ci + 1} of {n_chains} ===")

        if len(hops) >= 1:
            h1 = hops[0]
            title1 = h1.get("title", "").strip()
            text1 = h1.get("text", "").strip()
            parts.append(f"Bridge document: {title1}")
            if text1:
                parts.append(text1)
            parts.append("")

        if len(hops) >= 2:
            h2 = hops[1]
            title2 = h2.get("title", "").strip()
            text2 = h2.get("text", "").strip()
            parts.append(f"Answer document: {title2}")
            if text2:
                parts.append(text2)
            parts.append("")

    s = "\n".join(parts).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rsplit("\n", 1)[0]
    return s


# ─────────────────────────── answer extraction ──────────────────────

def extract_final(raw: str) -> str:
    """
    Qwen prefill path: assistant turn starts with '<final>', so vLLM
    returns only the content AFTER that prefix (the stop token '</final>'
    is consumed and not included in the response).
    """
    if not raw:
        return ""

    # Defensive: strip any residual thinking block
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    # Case 1: model echoed full <final>...</final> tags
    matches = FINAL_RE.findall(raw)
    if matches:
        ans = (matches[-1] or "").strip()
        lines = [l.strip() for l in ans.splitlines() if l.strip()]
        return " ".join(lines).strip() if lines else ""

    # Case 2: prefill path — raw IS the answer text
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def is_bad(ans: str) -> bool:
    a = ans.strip()
    if not a:
        return True
    low = a.lower()
    if low.startswith("[chain"):
        return True
    if "if the evidence does not contain" in low:
        return True
    if low.startswith("the evidence provided"):
        return True
    if low.startswith(("okay,", "alright,", "so,")):
        return True
    if low in {"unknown", "unk"}:
        return True
    if len(a) > 120:
        return True
    return False


# ─────────────────────────── vLLM client ────────────────────────────

class ExpectedN5Failure(RuntimeError):
    pass


def call_completions(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    timeout_s: int,
    n: int,
    stop: Optional[List[str]],
) -> List[str]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": "<final>"},
        ],
        "n":           int(n),
        "temperature": float(temperature),
        "top_p":       float(top_p),
        "max_tokens":  int(max_tokens),
        "seed":        int(seed),
    }
    if stop:
        payload["stop"] = stop

    data = json.dumps(payload).encode("utf-8")
    req  = request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body_text[:500]}") from e

    choices = body.get("choices", [])
    return [c["message"]["content"] for c in choices]


def retry_call(fn, retries: int = 3, sleep_s: float = 2.0):
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  [retry {attempt+1}/{retries}] {e}")
            time.sleep(sleep_s * (attempt + 1))


# ─────────────────────────── main ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Exp3: chain-aware prompting candidate generation")
    ap.add_argument("--evidence",       required=True,
                    help="exp1b/evidence/dev_K100_chains.jsonl")
    ap.add_argument("--gold",           required=True,
                    help="data/hotpotqa/raw/hotpot_dev_distractor_v1.json")
    ap.add_argument("--prompt_file",    required=True,
                    help="exp3/inputs/prompt_v3_chain_aware.txt")
    ap.add_argument("--prompt_version", default="v3_chain_aware")
    ap.add_argument("--evidence_format", choices=["flat", "chain_aware"],
                    default="chain_aware",
                    help="Evidence formatting strategy (default: chain_aware)")
    ap.add_argument("--out_jsonl",      required=True,
                    help="exp3/candidates/dev_M5_candidates_chain_aware.jsonl")
    ap.add_argument("--manifest",       required=True,
                    help="exp3/manifest.json")
    ap.add_argument("--split",          default="dev")
    ap.add_argument("--m",              type=int, default=5)
    ap.add_argument("--chains_for_llm", type=int, default=5,
                    help="Number of top chains to show the LLM (default: 5)")
    ap.add_argument("--max_evidence_chars", type=int, default=6000)
    ap.add_argument("--llm_base_url",   default="http://127.0.0.1:8000/v1")
    ap.add_argument("--llm_model_id",   default="qwen2.5-1.5b-instruct")
    ap.add_argument("--temperature",    type=float, default=0.7)
    ap.add_argument("--top_p",          type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int,   default=128)
    ap.add_argument("--seed",           type=int,   default=12345)
    ap.add_argument("--timeout_s",      type=int,   default=60)
    ap.add_argument("--http_retries",   type=int,   default=3)
    ap.add_argument("--http_retry_sleep", type=float, default=2.0)
    ap.add_argument("--resume",         action="store_true")
    ap.add_argument("--force_seed_loop", action="store_true",
                    help="Skip n=5 and always use seed+i fallback")
    args = ap.parse_args()

    # ── select evidence formatter ──
    if args.evidence_format == "chain_aware":
        format_evidence = evidence_text_chain_aware
        print(f"[exp3] Evidence format: CHAIN-AWARE")
    else:
        format_evidence = evidence_text_flat
        print(f"[exp3] Evidence format: FLAT (exp1b baseline)")

    # ── load prompt template ──
    prompt_t = read_text(args.prompt_file)
    prompt_hash = hashlib.sha256(prompt_t.encode()).hexdigest()
    print(f"[exp3] Prompt: {args.prompt_file}  (sha256={prompt_hash[:16]}...)")
    print(f"[exp3] Model:  {args.llm_model_id}")
    print(f"[exp3] M={args.m}  T={args.temperature}  top_p={args.top_p}  "
          f"seed={args.seed}")

    # ── resume support ──
    done: Set[str] = set()
    if args.resume:
        done = load_done_qids(args.out_jsonl)
        print(f"[exp3] Resuming: {len(done)} already done")

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    # ── evidence hash ──
    ev_sha = sha256_file(args.evidence)

    STOP_TOKENS = ["</final>"]

    # ── manifest ──
    update_manifest(args.manifest, {
        "exp3_generate_candidates": {
            "schema_version":  SCHEMA_VERSION,
            "model":           args.llm_model_id,
            "split":           args.split,
            "evidence_path":   args.evidence,
            "evidence_sha256": ev_sha,
            "evidence_format": args.evidence_format,
            "prompt_version":  args.prompt_version,
            "prompt_file":     os.path.abspath(args.prompt_file),
            "prompt_hash":     prompt_hash,
            "llm_base_url":    args.llm_base_url,
            "llm_model_id":    args.llm_model_id,
            "decoding": {
                "temperature":    args.temperature,
                "top_p":          args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "stop":           STOP_TOKENS,
                "prefill":        "<final>",
            },
            "seed":            args.seed,
            "chains_for_llm":  args.chains_for_llm,
            "started_at_utc":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "resume":          bool(args.resume),
            "strategy":        "try_n=5_then_fallback_seed+i",
            "endpoint":        "chat/completions",
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

        fixed_chains    = all_chains[:args.chains_for_llm]
        shown_chain_ids = [int(ch.get("chain_id", i))
                           for i, ch in enumerate(fixed_chains)]

        # ── THE CHANGE: use selected evidence formatter ──
        ev_txt = format_evidence(fixed_chains, args.max_evidence_chars)
        prompt = prompt_t.format(question=q, evidence=ev_txt)

        sampling_strategy = "n=5"
        answers: List[str] = []

        try:
            if args.force_seed_loop:
                raise ExpectedN5Failure("forced via --force_seed_loop")

            def _n5():
                return call_completions(
                    args.llm_base_url, args.llm_model_id, prompt,
                    args.temperature, args.top_p, args.max_new_tokens,
                    args.seed, args.timeout_s, n=args.m, stop=STOP_TOKENS,
                )

            raws = retry_call(_n5, retries=args.http_retries,
                              sleep_s=args.http_retry_sleep)

            if len(raws) != args.m:
                raise ExpectedN5Failure(
                    f"server returned {len(raws)} choices, expected {args.m}")

            answers = [extract_final(r) for r in raws]

            if len(set(answers)) == 1:
                raise ExpectedN5Failure(
                    "n=5 returned identical outputs — no diversity")

            empty_under_n5 = sum(1 for a in answers if not a.strip())
            if empty_under_n5 >= (args.m // 2 + 1):
                raise ExpectedN5Failure(
                    f"n=5 produced {empty_under_n5}/{args.m} empty answers")

            n5_ok += 1

        except ExpectedN5Failure as e:
            n5_fallback += 1
            if (n5_fallback <= 10) or (n5_fallback % 500 == 0):
                print(f"[exp3] qid={qid} fallback seed+i: {e}")
            sampling_strategy = "seed+i"
            answers = []
            for i in range(args.m):
                def _n1(i=i):
                    return call_completions(
                        args.llm_base_url, args.llm_model_id, prompt,
                        args.temperature, args.top_p, args.max_new_tokens,
                        args.seed + i, args.timeout_s, n=1, stop=STOP_TOKENS,
                    )
                raws = retry_call(_n1, retries=args.http_retries,
                                  sleep_s=args.http_retry_sleep)
                a = extract_final(raws[0]) if raws else ""
                answers.append(a)

        except Exception as e:
            raise RuntimeError(f"Unexpected error for qid={qid}: {e}") from e

        bad_count    = sum(1 for a in answers if is_bad(a))
        unique_count = len(set(answers))
        empty_count  = sum(1 for a in answers if not a.strip())

        rec = {
            "schema_version": SCHEMA_VERSION,
            "split":    args.split,
            "qid":      qid,
            "candidates": [
                {"answer_id": i, "answer_text": answers[i]}
                for i in range(args.m)
            ],
            "generation_context": {
                "sampling_strategy":   sampling_strategy,
                "chains_shown_to_llm": shown_chain_ids,
                "evidence_format":     args.evidence_format,
                "prompt_version":      args.prompt_version,
                "prompt_hash":         prompt_hash,
                "llm_base_url":        args.llm_base_url,
                "llm_model_id":        args.llm_model_id,
                "decoding": {
                    "temperature":    args.temperature,
                    "top_p":          args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "stop":           STOP_TOKENS,
                    "prefill":        "<final>",
                },
                "seed":              args.seed,
            },
            "stats": {
                "bad_count":    bad_count,
                "unique_count": unique_count,
                "empty_count":  empty_count,
            },
        }

        append_jsonl(args.out_jsonl, rec)
        wrote += 1

        if wrote % 200 == 0:
            elapsed_pct = (wrote + skipped) / 7405 * 100
            print(f"[exp3] progress: {wrote} wrote  {skipped} skipped  "
                  f"n5_ok={n5_ok} fallback={n5_fallback}  "
                  f"({elapsed_pct:.1f}%)")

    # ── done ──
    print(f"\n[exp3] DONE: wrote={wrote}  skipped={skipped}  "
          f"n5_ok={n5_ok}  fallback={n5_fallback}")

    update_manifest(args.manifest, {
        "exp3_generate_candidates_done": {
            "out_sha256":     sha256_file(args.out_jsonl) if wrote > 0 else None,
            "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wrote":          wrote,
            "skipped":        skipped,
            "n5_ok":          n5_ok,
            "fallback":       n5_fallback,
        }
    })


if __name__ == "__main__":
    main()