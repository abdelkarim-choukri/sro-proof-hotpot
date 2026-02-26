#!/usr/bin/env python3
"""
exp1_generate_candidates.py  —  Qwen2.5-1.5B-Instruct edition
Changes vs DeepSeek-R1 version:
  1. Assistant prefill: "<final>" only  (no <think> block)
  2. STOP_TOKENS: ["</final>"] only     (drop "\n\n" — was the root-cause bug)
  3. extract_final: Case 2 path handles prefill response directly (no change needed)
  4. max_new_tokens default: 128        (no reasoning block → short answers only)
  5. SCHEMA_VERSION bumped to v4_qwen
"""
import argparse, hashlib, json, os, time, re
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib import request, error

SCHEMA_VERSION = "sro-proof.exp1.candidates.v4_qwen"

FINAL_RE = re.compile(r"<final>\s*(.*?)\s*(?:</final>|\Z)", re.DOTALL | re.IGNORECASE)

# ---------------- utils ----------------
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

# -------------- prompt/evidence --------------
def evidence_text(chains: List[Dict[str, Any]], max_chars: int) -> str:
    parts: List[str] = []
    for ch in chains:
        cid = ch.get("chain_id", "?")
        for hop in ch.get("hops", []):
            parts.append(
                f"[chain {cid} hop {hop.get('hop','?')}] {hop.get('title','')}: {hop.get('text','')}"
            )
        parts.append("")
    s = "\n".join(parts).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rsplit("\n", 1)[0]
    return s

def extract_final(raw: str) -> str:
    """
    Qwen prefill path: assistant turn starts with '<final>', so vLLM
    returns only the content AFTER that prefix (the stop token '</final>'
    is consumed and not included in the response).

    Example raw response: 'Darren Aronofsky'
    No <final> tag in raw → falls through to Case 2 (first non-empty line).
    """
    if not raw:
        return ""

    # Defensive: strip any residual thinking block (shouldn't appear for Qwen)
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    # Case 1: model echoed full <final>...</final> tags (shouldn't happen with prefill)
    matches = FINAL_RE.findall(raw)
    if matches:
        ans = (matches[-1] or "").strip()
        lines = [l.strip() for l in ans.splitlines() if l.strip()]
        return " ".join(lines).strip() if lines else ""

    # Case 2: prefill path — raw IS the answer text, stop token already removed.
    # Take only the first non-empty line.
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
    if low.startswith("okay,") or low.startswith("alright,") or low.startswith("so,"):
        return True
    if low in {"unknown", "unk"}:
        return True
    if len(a) > 120:
        return True
    return False

# -------------- vLLM client --------------
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
            # CHANGE 1 (vs DeepSeek): prefill "<final>" only — no <think> block.
            # Qwen2.5-Instruct has no reasoning mode; injecting </think> would
            # confuse the model and produce garbage output.
            {"role": "assistant", "content": "<final>"},
        ],
        "n":           int(n),
        "temperature": float(temperature),
        "top_p":       float(top_p),
        "max_tokens":  int(max_tokens),
        "seed":        int(seed),
    }
    # CHANGE 2 (vs DeepSeek): stop = ["</final>"] ONLY.
    # The original bug was "\n\n" in stop — DeepSeek emits "\n\n" before <final>,
    # so the stop triggered before the answer was written. Qwen doesn't have this
    # problem, but we still drop "\n\n" to be safe.
    if stop:
        payload["stop"] = stop

    data = json.dumps(payload).encode("utf-8")
    req  = request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ExpectedN5Failure(f"HTTPError {e.code}: {body}")
    except Exception as e:
        raise RuntimeError(f"Network/IO error calling {url}: {e}")

    choices = out.get("choices", [])
    return [(c.get("message", {}).get("content", "") or "") for c in choices]

def retry_call(fn, retries: int, sleep_s: float):
    last = None
    for t in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            if t < retries:
                time.sleep(sleep_s * (2 ** t))
            else:
                raise last

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",            required=True, choices=["train", "dev", "test"])
    ap.add_argument("--evidence",         required=True)
    ap.add_argument("--m",                type=int,   default=5)
    ap.add_argument("--chains_for_llm",   type=int,   default=8)
    ap.add_argument("--max_evidence_chars", type=int, default=10000)

    ap.add_argument("--prompt_version",   required=True)
    ap.add_argument("--prompt_file",      required=True)

    ap.add_argument("--llm_base_url",     required=True)
    ap.add_argument("--llm_model_id",     required=True)

    ap.add_argument("--temperature",      type=float, required=True)
    ap.add_argument("--top_p",            type=float, required=True)

    # CHANGE 3 (vs DeepSeek): default 128, not 2048.
    # Qwen has no thinking block — HotpotQA answers are 1-5 words.
    # 128 tokens is plenty and keeps throughput high.
    ap.add_argument("--max_new_tokens",   type=int,   default=128)
    ap.add_argument("--seed",             type=int,   default=12345)
    ap.add_argument("--timeout_s",        type=int,   default=60)

    ap.add_argument("--phase0_record",    required=True)
    ap.add_argument("--phase0_versions",
                    default="/var/tmp/%s/sro/logs/phase0_versions.txt" % os.environ.get("USER", ""))
    ap.add_argument("--serve_cmd_file",
                    default="/var/tmp/%s/sro/logs/vllm_serve_cmd.txt" % os.environ.get("USER", ""))

    ap.add_argument("--out",              required=True)
    ap.add_argument("--out_sha256",       required=True)
    ap.add_argument("--manifest",         required=True)
    ap.add_argument("--resume",           action="store_true")
    ap.add_argument("--force_seed_loop",  action="store_true")
    ap.add_argument("--http_retries",     type=int,   default=2)
    ap.add_argument("--http_retry_sleep", type=float, default=1.0)

    args = ap.parse_args()

    if args.temperature <= 0.0:
        raise ValueError("temperature must be > 0 for sampling diversity.")

    prompt_t    = read_text(args.prompt_file)
    prompt_hash = hashlib.sha256(prompt_t.encode("utf-8")).hexdigest()
    ev_sha      = sha256_file(args.evidence)
    phase0      = load_json(args.phase0_record)

    vllm_version = torch_version = None
    if os.path.exists(args.phase0_versions):
        for line in read_text(args.phase0_versions).splitlines():
            if line.startswith("vllm "):  vllm_version = line.split(" ", 1)[1]
            if line.startswith("torch "): torch_version = line.split(" ", 1)[1]
    serve_cmd = (read_text(args.serve_cmd_file).strip()
                 if os.path.exists(args.serve_cmd_file) else None)

    done = load_done_qids(args.out) if args.resume else set()

    # CHANGE 4 (vs DeepSeek): ONLY "</final>" in stop list.
    STOP_TOKENS = ["</final>"]

    update_manifest(args.manifest, {
        "exp1_step2_candidates_qwen": {
            "schema_version": SCHEMA_VERSION,
            "model":          "qwen2.5-1.5b-instruct",
            "split":          args.split,
            "evidence_path":  args.evidence,
            "evidence_sha256": ev_sha,
            "prompt_version": args.prompt_version,
            "prompt_file":    args.prompt_file,
            "prompt_hash":    prompt_hash,
            "llm_base_url":   args.llm_base_url,
            "llm_model_id":   args.llm_model_id,
            "decoding": {
                "temperature":  args.temperature,
                "top_p":        args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "stop":         STOP_TOKENS,
                "prefill":      "<final>",
            },
            "seed":           args.seed,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "resume":         bool(args.resume),
            "strategy":       "try_n=5_then_fallback_seed+i",
            "endpoint":       "chat/completions",
        }
    })

    wrote = skipped = n5_ok = n5_fallback = 0

    for evd in iter_jsonl(args.evidence):
        qid = str(evd["qid"])
        if qid in done:
            skipped += 1
            continue

        q          = evd["question"]
        all_chains = evd.get("chains", [])

        fixed_chains    = all_chains[:args.chains_for_llm]
        shown_chain_ids = [int(ch.get("chain_id", i)) for i, ch in enumerate(fixed_chains)]

        ev_txt = evidence_text(fixed_chains, args.max_evidence_chars)
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

            raws = retry_call(_n5, retries=args.http_retries, sleep_s=args.http_retry_sleep)

            if len(raws) != args.m:
                raise ExpectedN5Failure(f"server returned {len(raws)} choices, expected {args.m}")

            answers = [extract_final(r) for r in raws]

            if len(set(answers)) == 1:
                raise ExpectedN5Failure("n=5 returned identical outputs — no diversity")

            empty_under_n5 = sum(1 for a in answers if not a.strip())
            if empty_under_n5 >= (args.m // 2 + 1):
                raise ExpectedN5Failure(f"n=5 produced {empty_under_n5}/{args.m} empty answers")

            n5_ok += 1

        except ExpectedN5Failure as e:
            n5_fallback += 1
            print(f"[step2v4] qid={qid} fallback seed+i: {e}")
            sampling_strategy = "seed+i"
            answers = []
            for i in range(args.m):
                def _n1(i=i):
                    return call_completions(
                        args.llm_base_url, args.llm_model_id, prompt,
                        args.temperature, args.top_p, args.max_new_tokens,
                        args.seed + i, args.timeout_s, n=1, stop=STOP_TOKENS,
                    )
                raws = retry_call(_n1, retries=args.http_retries, sleep_s=args.http_retry_sleep)
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
                "sampling_strategy":  sampling_strategy,
                "chains_shown_to_llm": shown_chain_ids,
                "prompt_version":     args.prompt_version,
                "prompt_hash":        prompt_hash,
                "llm_base_url":       args.llm_base_url,
                "llm_model_id":       args.llm_model_id,
                "decoding": {
                    "temperature":    args.temperature,
                    "top_p":          args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "stop":           STOP_TOKENS,
                    "prefill":        "<final>",
                },
                "seed": args.seed,
                "stats": {
                    "bad_count":    bad_count,
                    "unique_count": unique_count,
                    "empty_count":  empty_count,
                },
                "provenance": {
                    "vllm_version":    vllm_version,
                    "torch_version":   torch_version,
                    "serve_cmd":       serve_cmd,
                    "snapshot_sha256": phase0.get("snapshot_sha256"),
                    "snapshot_path":   phase0.get("snapshot_path"),
                    "hf_commit":       phase0.get("hf_commit"),
                },
            },
            "source": {
                "evidence_path":   args.evidence,
                "evidence_sha256": ev_sha,
            },
        }

        append_jsonl(args.out, rec)
        wrote += 1
        if wrote % 100 == 0:
            print(f"[step2v4] wrote={wrote} skipped={skipped} "
                  f"n5_ok={n5_ok} fallback={n5_fallback}")

    write_sha256_file(args.out, args.out_sha256)
    update_manifest(args.manifest, {
        "exp1_step2_candidates_qwen_done": {
            "out_sha256":    sha256_file(args.out),
            "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wrote":    wrote,
            "skipped":  skipped,
            "n5_ok":    n5_ok,
            "fallback": n5_fallback,
        }
    })
    print(f"[step2v4] DONE wrote={wrote} skipped={skipped} "
          f"n5_ok={n5_ok} fallback={n5_fallback}")
    print(f"[step2v4] out={args.out}")

if __name__ == "__main__":
    main()