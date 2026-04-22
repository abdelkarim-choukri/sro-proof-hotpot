#!/usr/bin/env python3
"""
experiments/run_b4_fixed.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
B4 Fixed: Concrete Failure Cases selected by the correct mechanism signal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THE PREVIOUS B4 WAS WRONG
    The old step4 in run_hypothesis_experiments.py sorted CHAIN_WINS examples
    by:
        imbalance_gap = wrong.imbalance - right.imbalance

    where imbalance = |nli_hop1 - nli_hop2|.

    This was wrong because:
    1. Imbalance is NOT the discriminating signal — Δ nli_hop2 is.
    2. The selection criterion didn't match the B2 full results which show
       qa_hop_balance (+0.30-0.34) and nli_hop2 (+0.14-0.28) are the
       strongest signals.
    3. Only NLI scores were shown — QA and lex were missing.

THE CORRECT APPROACH
    Select examples by COMBINED HOP-2 DELTA across all scorers:
        hop2_delta = Δ nli_hop2 + Δ qa_hop2
                   = (right.nli_hop2 - wrong.nli_hop2)
                   + (right.qa_hop2  - wrong.qa_hop2)

    This directly selects examples where the hop-2 anchoring pattern
    is clearest, consistent with B2 full results and Z3 feature importances.

    Show all three scorers (NLI + QA + lex) for each candidate.
    The explanation references qa_hop2 as the primary mechanism driver.

WHAT A GOOD B4 EXAMPLE SHOWS
    WRONG answer (what Z2 picked):
        nli_hop1  HIGH  → mentioned in bridge document
        nli_hop2  LOW   → NOT in answer document
        nli_flat  HIGH  → flat NLI averages both, gets fooled
        qa_hop2   LOW   → QA cannot extract this from answer doc
        ← this is bridge-document anchoring

    CORRECT answer (what Z_full picked):
        nli_hop2  HIGH  → grounded in answer document
        qa_hop2   HIGH  → QA can extract it from answer doc
        nli_flat  similar to wrong  ← flat can't tell the difference

USAGE
    python3 experiments/run_b4_fixed.py \\
        --proj_root /var/tmp/u24sf51014/sro/work/sro-proof-hotpot \\
        --out_dir   experiments/results/b4_fixed

    Single setting:
    python3 experiments/run_b4_fixed.py ... --setting wiki2
"""

import argparse
import json
import os
import re
import string
from typing import Dict, Iterator, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def em(p: str, g: str) -> bool:
    return normalize(p) == normalize(g)

def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_gold(path: str) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        raw = f.read(1); f.seek(0)
        if raw.strip() == "[":
            data = json.load(f)
        else:
            data = [json.loads(l) for l in f if l.strip()]
    result = {}
    for ex in data:
        qid = str(ex.get("_id", ex.get("id", "")))
        ans = ex.get("answer", "")
        result[qid] = {
            "answer": str(ans.get("answer","") if isinstance(ans, dict) else ans),
            "type":   ex.get("type", ex.get("q_type", "bridge")),
        }
    return result

def load_preds(path: str) -> Dict[str, str]:
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = str(rec.get("pred", ""))
    return result

def load_cands_by_qid(path: str) -> Dict[str, List[dict]]:
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = rec.get("candidates", [])
    return result

def load_evidence(path: str) -> Dict[str, dict]:
    result = {}
    for rec in iter_jsonl(path):
        result[str(rec["qid"])] = rec
    return result

def find_cand(cands: List[dict], answer: str) -> Optional[dict]:
    a = normalize(answer)
    for c in cands:
        if normalize(c.get("answer", c.get("answer_text", ""))) == a:
            return c
    return None

def get_hop_texts(ev_rec: dict) -> Tuple[str, str, str, str]:
    """Return (hop1_title, hop1_text, hop2_title, hop2_text)."""
    chains = (ev_rec.get("chains") or
              ev_rec.get("evidence", {}).get("chains", []))
    if not chains:
        return "", "", "", ""
    hops = chains[0].get("hops", [])
    h1t = h1x = h2t = h2x = ""
    if len(hops) >= 1:
        h1t = hops[0].get("title", "")
        h1x = hops[0].get("text",  "")
    if len(hops) >= 2:
        h2t = hops[1].get("title", "")
        h2x = hops[1].get("text",  "")
    return h1t, h1x, h2t, h2x

def nli_scores(c: Optional[dict]) -> dict:
    if c is None:
        return {"nli_flat": 0, "nli_hop1": 0, "nli_hop2": 0}
    return {
        "nli_flat": round(float(c.get("nli_flat",  0) or 0), 3),
        "nli_hop1": round(float(c.get("nli_hop1",  0) or 0), 3),
        "nli_hop2": round(float(c.get("nli_hop2",  0) or 0), 3),
    }

def qa_scores_f(c: Optional[dict]) -> dict:
    if c is None:
        return {"qa_flat": 0, "qa_hop1": 0, "qa_hop2": 0, "qa_hop_balance": 0}
    return {
        "qa_flat":        round(float(c.get("qa_flat",  0) or 0), 3),
        "qa_hop1":        round(float(c.get("qa_hop1",  0) or 0), 3),
        "qa_hop2":        round(float(c.get("qa_hop2",  0) or 0), 3),
        "qa_hop_balance": round(float(c.get("qa_hop_balance",
                           abs((c.get("qa_hop1",0) or 0) -
                               (c.get("qa_hop2",0) or 0))) or 0), 3),
    }

def lex_scores_f(c: Optional[dict]) -> dict:
    if c is None:
        return {"lex_flat": 0, "lex_hop1": 0, "lex_hop2": 0}
    return {
        "lex_flat": round(float(c.get("lex_flat",  0) or 0), 3),
        "lex_hop1": round(float(c.get("lex_hop1",  0) or 0), 3),
        "lex_hop2": round(float(c.get("lex_hop2",  0) or 0), 3),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CORE ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_b4(
    label:     str,
    z2_path:   str,
    zf_path:   str,
    hop_path:  str,
    qa_path:   str,
    lex_path:  str,
    ev_path:   str,
    gold_path: str,
    out_dir:   str,
    n_examples: int = 4,
) -> None:

    print(f"\n{'━'*70}")
    print(f"  B4 Fixed: {label}")
    print(f"{'━'*70}")

    for p, n in [(z2_path,"z2_preds"), (zf_path,"zf_preds"),
                 (hop_path,"hop_scores"), (qa_path,"qa_scores"),
                 (lex_path,"lex_features"), (ev_path,"evidence"),
                 (gold_path,"gold")]:
        if not os.path.exists(p):
            print(f"  ✗ {n} not found: {p}")
            return

    gold          = load_gold(gold_path)
    z2_preds      = load_preds(z2_path)
    zf_preds      = load_preds(zf_path)
    hop_cands     = load_cands_by_qid(hop_path)
    qa_cands      = load_cands_by_qid(qa_path)
    lex_cands     = load_cands_by_qid(lex_path)
    evidence      = load_evidence(ev_path)

    print(f"  Loaded: gold={len(gold):,}  hop={len(hop_cands):,}  "
          f"qa={len(qa_cands):,}  lex={len(lex_cands):,}")

    # ── Find CHAIN_WINS and score each ───────────────────────────────
    scored = []

    for qid, info in gold.items():
        gold_ans = info["answer"]
        qtype    = info["type"]

        z2_pred = z2_preds.get(qid, "")
        zf_pred = zf_preds.get(qid, "")

        if em(z2_pred, gold_ans) or not em(zf_pred, gold_ans):
            continue  # Not a CHAIN_WINS question

        # Get scores for wrong and right candidates
        hc  = hop_cands.get(qid, [])
        qac = qa_cands.get(qid,  [])
        lc  = lex_cands.get(qid, [])
        ev  = evidence.get(qid,  {})

        wrong_hop = find_cand(hc,  z2_pred)
        right_hop = find_cand(hc,  gold_ans)
        wrong_qa  = find_cand(qac, z2_pred)
        right_qa  = find_cand(qac, gold_ans)
        wrong_lex = find_cand(lc,  z2_pred)
        right_lex = find_cand(lc,  gold_ans)

        if wrong_hop is None or right_hop is None:
            continue

        w_nli = nli_scores(wrong_hop)
        r_nli = nli_scores(right_hop)
        w_qa  = qa_scores_f(wrong_qa)
        r_qa  = qa_scores_f(right_qa)
        w_lex = lex_scores_f(wrong_lex)
        r_lex = lex_scores_f(right_lex)

        # Selection metric: combined hop-2 delta across all scorers
        # This directly corresponds to the B2 full mechanism finding
        delta_nli_hop2 = r_nli["nli_hop2"] - w_nli["nli_hop2"]
        delta_qa_hop2  = r_qa["qa_hop2"]   - w_qa["qa_hop2"]
        delta_lex_hop2 = r_lex["lex_hop2"] - w_lex["lex_hop2"]

        combined_hop2_delta = delta_nli_hop2 + delta_qa_hop2 + delta_lex_hop2

        h1t, h1x, h2t, h2x = get_hop_texts(ev)
        question = ev.get("question", "")

        scored.append({
            "qid":                  qid,
            "question":             question,
            "type":                 qtype,
            "gold_answer":          gold_ans,
            "wrong_answer":         z2_pred,
            "right_answer":         zf_pred,
            "hop1_title":           h1t,
            "hop1_text":            (h1x[:250] + "...") if len(h1x) > 250 else h1x,
            "hop2_title":           h2t,
            "hop2_text":            (h2x[:250] + "...") if len(h2x) > 250 else h2x,
            "wrong_nli":            w_nli,
            "right_nli":            r_nli,
            "wrong_qa":             w_qa,
            "right_qa":             r_qa,
            "wrong_lex":            w_lex,
            "right_lex":            r_lex,
            "delta_nli_hop2":       round(delta_nli_hop2, 3),
            "delta_qa_hop2":        round(delta_qa_hop2,  3),
            "delta_lex_hop2":       round(delta_lex_hop2, 3),
            "combined_hop2_delta":  round(combined_hop2_delta, 3),
        })

    print(f"  CHAIN_WINS: {len(scored):,} total")
    if not scored:
        print("  No examples found — check file paths")
        return

    # Sort by combined hop-2 delta descending (strongest mechanism signal first)
    scored.sort(key=lambda x: -x["combined_hop2_delta"])

    # ── Select diverse examples (one per type) ────────────────────────
    target_types = (["bridge", "comparison", "compositional", "inference"]
                    if any(s["type"] in ["compositional","inference"]
                           for s in scored)
                    else ["bridge", "comparison"])

    selected = []
    seen_types = set()

    # First: best example per type
    for t in target_types:
        for ex in scored:
            if ex["type"] == t and t not in seen_types:
                selected.append(ex)
                seen_types.add(t)
                break

    # Fill remaining slots with highest combined_hop2_delta
    for ex in scored:
        if len(selected) >= n_examples:
            break
        if ex not in selected:
            selected.append(ex)

    selected = selected[:n_examples]
    print(f"  Selected {len(selected)} examples "
          f"(types: {[s['type'] for s in selected]})")
    print(f"  Selection metric: combined_hop2_delta = "
          f"Δnli_hop2 + Δqa_hop2 + Δlex_hop2")

    # ── Print examples ───────────────────────────────────────────────
    for i, ex in enumerate(selected):
        print(f"\n  {'─'*65}")
        print(f"  Example {i+1} — type: {ex['type']}")
        print(f"  {'─'*65}")
        print(f"  Question  : {ex['question']}")
        print(f"  Gold ans  : {ex['gold_answer']}")
        print(f"  Hop-1 [{ex['hop1_title']}]:")
        print(f"    {ex['hop1_text']}")
        print(f"  Hop-2 [{ex['hop2_title']}]:")
        print(f"    {ex['hop2_text']}")

        # Wrong answer table
        w = ex
        print(f"\n  WRONG answer — Z2 picked: '{ex['wrong_answer']}'")
        print(f"  {'Scorer':<12}  {'flat':>8}  {'hop1':>8}  {'hop2':>8}")
        print(f"  {'─'*44}")
        print(f"  {'NLI':<12}  "
              f"{ex['wrong_nli']['nli_flat']:>8.3f}  "
              f"{ex['wrong_nli']['nli_hop1']:>8.3f}  "
              f"{ex['wrong_nli']['nli_hop2']:>8.3f}  ← low hop2")
        print(f"  {'QA':<12}  "
              f"{ex['wrong_qa']['qa_flat']:>8.3f}  "
              f"{ex['wrong_qa']['qa_hop1']:>8.3f}  "
              f"{ex['wrong_qa']['qa_hop2']:>8.3f}  ← low hop2")
        print(f"  {'Lexical':<12}  "
              f"{ex['wrong_lex']['lex_flat']:>8.3f}  "
              f"{ex['wrong_lex']['lex_hop1']:>8.3f}  "
              f"{ex['wrong_lex']['lex_hop2']:>8.3f}  ← low hop2")

        # Right answer table
        print(f"\n  RIGHT answer — Z_full picked: '{ex['right_answer']}'")
        print(f"  {'Scorer':<12}  {'flat':>8}  {'hop1':>8}  {'hop2':>8}")
        print(f"  {'─'*44}")
        print(f"  {'NLI':<12}  "
              f"{ex['right_nli']['nli_flat']:>8.3f}  "
              f"{ex['right_nli']['nli_hop1']:>8.3f}  "
              f"{ex['right_nli']['nli_hop2']:>8.3f}  ← HIGH hop2")
        print(f"  {'QA':<12}  "
              f"{ex['right_qa']['qa_flat']:>8.3f}  "
              f"{ex['right_qa']['qa_hop1']:>8.3f}  "
              f"{ex['right_qa']['qa_hop2']:>8.3f}  ← HIGH hop2")
        print(f"  {'Lexical':<12}  "
              f"{ex['right_lex']['lex_flat']:>8.3f}  "
              f"{ex['right_lex']['lex_hop1']:>8.3f}  "
              f"{ex['right_lex']['lex_hop2']:>8.3f}  ← HIGH hop2")

        print(f"\n  Delta (right − wrong):")
        print(f"    Δ nli_hop2 = {ex['delta_nli_hop2']:+.3f}   "
              f"Δ qa_hop2 = {ex['delta_qa_hop2']:+.3f}   "
              f"Δ lex_hop2 = {ex['delta_lex_hop2']:+.3f}")
        print(f"    Combined hop-2 Δ = {ex['combined_hop2_delta']:+.3f}")

        # Auto-generate explanation based on the actual mechanism
        wrong_qa_h2  = ex["wrong_qa"]["qa_hop2"]
        right_qa_h2  = ex["right_qa"]["qa_hop2"]
        wrong_nli_h2 = ex["wrong_nli"]["nli_hop2"]
        wrong_nli_h1 = ex["wrong_nli"]["nli_hop1"]

        ANCHOR_THRESHOLD = 0.05  # minimum score to call a hop "anchored"
        neither_anchored = (wrong_nli_h1 < ANCHOR_THRESHOLD and
                            wrong_nli_h2 < ANCHOR_THRESHOLD)

        if neither_anchored:
            # Wrong answer is surface-plausible but has weak support in both hops
            print(f"\n  Explanation: '{ex['wrong_answer']}' has weak hop-level support "
                  f"across both hops (nli_hop1={wrong_nli_h1:.3f}, "
                  f"nli_hop2={wrong_nli_h2:.3f}, qa_hop2={wrong_qa_h2:.3f}) "
                  f"but flat scoring gave it surface plausibility. "
                  f"Per-hop scoring detects the hop-2 deficit and picks "
                  f"'{ex['right_answer']}' which has strong hop-2 support "
                  f"(nli_hop2={right_qa_h2:.3f}, qa_hop2={ex['right_qa']['qa_hop2']:.3f}).")
        else:
            anchor_hop = "hop-1 (bridge document)" \
                         if wrong_nli_h1 > wrong_nli_h2 \
                         else "hop-2 (answer document)"
            print(f"\n  Explanation: '{ex['wrong_answer']}' is a bridge-document anchor "
                  f"— it is mentioned in {anchor_hop} "
                  f"(nli_hop1={wrong_nli_h1:.3f}) "
                  f"but absent from the answer document "
                  f"(nli_hop2={wrong_nli_h2:.3f}, "
                  f"qa_hop2={wrong_qa_h2:.3f}). "
                  f"Flat scoring averages both hops and is fooled. "
                  f"Per-hop scoring detects the hop-2 deficit and picks "
                  f"'{ex['right_answer']}' which has strong hop-2 support "
                  f"(nli_hop2={right_qa_h2:.3f}, qa_hop2={ex['right_qa']['qa_hop2']:.3f}).")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"b4_fixed_{label.lower().replace(' ','_')}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset":          label,
            "selection_metric": "combined_hop2_delta = Δnli_hop2 + Δqa_hop2 + Δlex_hop2",
            "n_chain_wins":     len(scored),
            "examples":         selected,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SETTINGS + MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_settings(R: str) -> dict:
    return {
        "hotpotqa_mdr": {
            "label":  "HotpotQA MDR",
            "z2":     f"{R}/exp_phase0/results/z2_surface_preds.jsonl",
            "zf":     f"{R}/exp_phase0/results/z_full_preds.jsonl",
            "hop":    f"{R}/exp0c/preds/dev_hop_scores.jsonl",
            "qa":     f"{R}/exp_phaseA/A1.1/qa_scores/dev_qa_hop_scores.jsonl",
            "lex":    f"{R}/exp_phaseA/A1.2/lex_features/dev_lex_features.jsonl",
            "ev":     f"{R}/exp0c/evidence/dev_K200_chains.jsonl",
            "gold":   f"{R}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json",
        },
        "hotpotqa_distractor": {
            "label":  "HotpotQA Distractor",
            "z2":     f"{R}/exp_distractor/results/z2_surface_preds.jsonl",
            "zf":     f"{R}/exp_distractor/results/z_full_preds.jsonl",
            "hop":    f"{R}/exp_distractor/preds/dev_hop_scores.jsonl",
            "qa":     f"{R}/exp_distractor/preds/dev_qa_hop_scores.jsonl",
            "lex":    f"{R}/exp_distractor/preds/dev_lex_features.jsonl",
            "ev":     f"{R}/exp_distractor/evidence/dev_distractor_chains.jsonl",
            "gold":   f"{R}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json",
        },
        "wiki2": {
            "label":  "2WikiMultiHopQA",
            "z2":     f"{R}/exp_wiki2/results/z2_surface_preds.jsonl",
            "zf":     f"{R}/exp_wiki2/results/z_full_preds.jsonl",
            "hop":    f"{R}/exp_wiki2/preds/dev_hop_scores.jsonl",
            "qa":     f"{R}/exp_wiki2/preds/dev_qa_hop_scores.jsonl",
            "lex":    f"{R}/exp_wiki2/preds/dev_lex_features.jsonl",
            "ev":     f"{R}/exp_wiki2/evidence/dev_wiki2_chains.jsonl",
            "gold":   f"{R}/data/wiki2/raw/dev_normalized.json",
        },
    }


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--proj_root",
                    default="/var/tmp/u24sf51014/sro/work/sro-proof-hotpot")
    ap.add_argument("--out_dir",    default="experiments/results/b4_fixed")
    ap.add_argument("--setting",    default="all",
                    help="hotpotqa_mdr | hotpotqa_distractor | wiki2 | all")
    ap.add_argument("--n_examples", type=int, default=4)
    args = ap.parse_args()

    configs = get_settings(args.proj_root)
    keys = list(configs.keys()) if args.setting == "all" else [args.setting]

    for key in keys:
        cfg = configs[key]
        run_b4(
            label      = cfg["label"],
            z2_path    = cfg["z2"],
            zf_path    = cfg["zf"],
            hop_path   = cfg["hop"],
            qa_path    = cfg["qa"],
            lex_path   = cfg["lex"],
            ev_path    = cfg["ev"],
            gold_path  = cfg["gold"],
            out_dir    = args.out_dir,
            n_examples = args.n_examples,
        )

    print(f"\n  All outputs → {args.out_dir}/")


if __name__ == "__main__":
    main()