#!/usr/bin/env python3
import argparse, json, os, re, string, hashlib, time
from collections import Counter
from typing import Dict, Any, List, Tuple

SCHEMA_SUMMARY = "sro-proof.exp1.oracle_summary.v1"
SCHEMA_PERQID  = "sro-proof.exp1.oracle_perqid.v1"

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def normalize_answer(s: str) -> str:
    # Hotpot/SQuAD-style
    def lower(text): return text.lower()
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)

def em_score(pred: str, gold: str) -> int:
    return 1 if normalize_answer(pred) == normalize_answer(gold) else 0

def load_docrecall_subset(evidence_jsonl: str) -> Dict[str, int]:
    m = {}
    for j in iter_jsonl(evidence_jsonl):
        qid = str(j["qid"])
        # support both schema variants:
        # v_old: derived.doc_recall_union
        # v_new (exp1b): flags.doc_recall_at_k (True/False/None)
        val = (j.get("derived", {}).get("doc_recall_union")
               or j.get("flags",   {}).get("doc_recall_at_k"))
        m[qid] = 1 if val else 0
    return m

def load_gold_answers(evidence_jsonl: str) -> Dict[str, str]:
    m = {}
    for j in iter_jsonl(evidence_jsonl):
        qid = str(j["qid"])
        m[qid] = j["gold"]["answer"]
    return m

def load_candidates(cand_jsonl: str) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    for j in iter_jsonl(cand_jsonl):
        qid = str(j["qid"])
        cands = j.get("candidates", [])
        # ensure sorted by answer_id
        cands = sorted(cands, key=lambda x: x.get("answer_id", 0))
        m[qid] = [c.get("answer_text", "") for c in cands]
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train","dev","test"])
    ap.add_argument("--gold", required=False)  # not needed; we use evidence pack gold
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--subset_key", default="doc_recall_union")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_sha256", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    docrec = load_docrecall_subset(args.evidence)
    gold = load_gold_answers(args.evidence)
    cands = load_candidates(args.candidates)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    total_em = total_f1 = 0.0
    sub_em = sub_f1 = 0.0
    n_all = 0
    n_sub = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for qid, g in gold.items():
            cand_list = cands.get(qid, [])[:args.m]
            # pad if missing
            while len(cand_list) < args.m:
                cand_list.append("")
            per = []
            best_em = 0
            best_f1 = 0.0
            best_id = 0
            for i, a in enumerate(cand_list):
                em = em_score(a, g)
                f1 = f1_score(a, g)
                per.append({"answer_id": i, "em": em, "f1": f1})
                if (f1 > best_f1) or (f1 == best_f1 and em > best_em):
                    best_f1, best_em, best_id = f1, em, i

            rec = {
                "schema_version": SCHEMA_PERQID,
                "split": args.split,
                "qid": qid,
                "doc_recall_union": int(docrec.get(qid, 0)),
                "best_answer_id": best_id,
                "best_em": best_em,
                "best_f1": best_f1,
                "per_candidate": per
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n_all += 1
            total_em += best_em
            total_f1 += best_f1

            if int(docrec.get(qid, 0)) == 1:
                n_sub += 1
                sub_em += best_em
                sub_f1 += best_f1

    summary = {
        "schema_version": SCHEMA_SUMMARY,
        "split": args.split,
        "m": args.m,
        "overall": {
            "n": n_all,
            "oracle_em": total_em / n_all if n_all else 0.0,
            "oracle_f1": total_f1 / n_all if n_all else 0.0
        },
        "subset_docrecall1": {
            "n": n_sub,
            "oracle_em": sub_em / n_sub if n_sub else 0.0,
            "oracle_f1": sub_f1 / n_sub if n_sub else 0.0
        },
        "source": {
            "evidence_path": args.evidence,
            "candidates_path": args.candidates
        }
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    out_sha = sha256_file(args.out_json)
    with open(args.out_sha256, "w", encoding="utf-8") as f:
        f.write(out_sha + "  " + os.path.basename(args.out_json) + "\n")

    # manifest update (shallow merge)
    man = {}
    if os.path.exists(args.manifest):
        with open(args.manifest, "r", encoding="utf-8") as f:
            man = json.load(f)
    man["exp1_step3_oracle"] = {
        "split": args.split,
        "m": args.m,
        "out_json": args.out_json,
        "out_json_sha256": out_sha,
        "out_jsonl": args.out_jsonl,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
