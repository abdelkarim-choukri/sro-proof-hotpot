#!/usr/bin/env python3
"""
exp3b_nli_prepare_finetune_data.py — Build NLI fine-tuning data from HotpotQA train

Extracts (premise, hypothesis, label) pairs that match how the pipeline uses NLI:
  - Premise = a passage sentence (like hop1 or hop2 text)
  - Hypothesis = an answer string (short phrase)
  - Label = entailment (1) / neutral (2) / contradiction (0)

Data sources from each HotpotQA training example:
  ENTAILMENT pairs:
    - premise = each supporting fact sentence
    - hypothesis = gold answer
    - Rationale: these sentences directly support the answer

  HARD NEUTRAL pairs (same-title, non-supporting):
    - premise = sentence from a gold-title paragraph that is NOT a supporting fact
    - hypothesis = gold answer
    - Rationale: same topic but doesn't entail the answer — the hardest negatives

  EASY NEUTRAL pairs (distractor paragraphs):
    - premise = sentence from a distractor paragraph
    - hypothesis = gold answer
    - Rationale: unrelated content

  CONTRADICTION pairs (cross-answer):
    - premise = supporting fact sentence from question A
    - hypothesis = gold answer from a DIFFERENT question B
    - Rationale: teaches the model that evidence for one answer doesn't support another

Balance target: ~1:1:1 entailment:neutral:contradiction per batch
  (achieved by sampling negatives to match positive count)

Output: train.jsonl and val.jsonl with {premise, hypothesis, label} per line
  Val = 5% held-out from training questions (NOT the HotpotQA dev set)

Usage:
    python3 tools/exp3b_nli_prepare_finetune_data.py \
        --hotpot_train  data/hotpotqa/raw/hotpot_train_v1.1.json \
        --out_dir       exp3b/nli_finetune/data \
        --val_frac      0.05 \
        --max_neg_per_question 6 \
        --seed          42
"""

import argparse
import json
import os
import random
import re


def normalize_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").replace("_", " ")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot_train", required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--val_frac",     type=float, default=0.05)
    ap.add_argument("--max_neg_per_question", type=int, default=6,
                    help="Max neutral+contradiction pairs per question "
                         "(to balance with ~2-4 positives)")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[prep] Loading {args.hotpot_train} ...")
    data = json.load(open(args.hotpot_train))
    print(f"[prep] {len(data)} training examples loaded")

    # ── shuffle and split train/val at question level ──
    random.shuffle(data)
    n_val = max(1, int(len(data) * args.val_frac))
    val_data  = data[:n_val]
    train_data = data[n_val:]
    print(f"[prep] Train: {len(train_data)}  Val: {len(val_data)}")

    # ── collect all gold answers for cross-answer contradiction sampling ──
    all_answers = [ex["answer"] for ex in data if ex.get("answer")]

    def extract_pairs(examples, split_name):
        pairs = []
        n_entail = n_hard_neutral = n_easy_neutral = n_contra = 0
        n_skipped_no_sf = 0

        for ex in examples:
            answer = ex.get("answer", "").strip()
            if not answer:
                continue

            # Build context map: title → list of sentence strings
            ctx_map = {}
            for title, sents in ex.get("context", []):
                t = normalize_title(title)
                ctx_map[t] = sents  # list of str

            # Supporting facts: (title, sent_id) pairs
            sfs = ex.get("supporting_facts", [])
            sf_set = set()
            gold_titles = set()
            for pair in sfs:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    t = normalize_title(pair[0])
                    sid = int(pair[1])
                    sf_set.add((t, sid))
                    gold_titles.add(t)

            if not sf_set:
                n_skipped_no_sf += 1
                continue

            # ── ENTAILMENT: supporting fact sentences → gold answer ──
            entail_premises = []
            for (t, sid) in sf_set:
                sents = ctx_map.get(t, [])
                if sid < len(sents):
                    sent_text = sents[sid].strip()
                    if sent_text:
                        pairs.append({
                            "premise": sent_text,
                            "hypothesis": answer,
                            "label": 1,  # entailment
                        })
                        entail_premises.append(sent_text)
                        n_entail += 1

            n_pos = len(entail_premises)
            if n_pos == 0:
                continue

            # Budget for negatives: match positive count
            max_neg = min(args.max_neg_per_question, n_pos * 2)
            n_hard_budget = max_neg // 2
            n_easy_budget = max_neg - n_hard_budget

            # ── HARD NEUTRAL: same gold-title paragraph, non-SF sentence ──
            hard_candidates = []
            for t in gold_titles:
                sents = ctx_map.get(t, [])
                for sid, sent_text in enumerate(sents):
                    if (t, sid) not in sf_set and sent_text.strip():
                        hard_candidates.append(sent_text.strip())

            random.shuffle(hard_candidates)
            for sent in hard_candidates[:n_hard_budget]:
                pairs.append({
                    "premise": sent,
                    "hypothesis": answer,
                    "label": 2,  # neutral
                })
                n_hard_neutral += 1

            # ── EASY NEUTRAL: distractor paragraph sentences ──
            easy_candidates = []
            for t, sents in ctx_map.items():
                if t not in gold_titles:
                    for sent_text in sents:
                        if sent_text.strip():
                            easy_candidates.append(sent_text.strip())

            random.shuffle(easy_candidates)
            for sent in easy_candidates[:n_easy_budget]:
                pairs.append({
                    "premise": sent,
                    "hypothesis": answer,
                    "label": 2,  # neutral
                })
                n_easy_neutral += 1

            # ── CONTRADICTION: SF sentence → wrong answer ──
            # Pick a random other answer as hypothesis
            contra_budget = min(n_pos, 2)  # 1-2 per question
            if entail_premises and len(all_answers) > 10:
                wrong_answers = random.sample(all_answers,
                                              min(contra_budget * 3, len(all_answers)))
                wrong_answers = [a for a in wrong_answers
                                 if a.lower().strip() != answer.lower().strip()][:contra_budget]
                for wrong_ans in wrong_answers:
                    prem = random.choice(entail_premises)
                    pairs.append({
                        "premise": prem,
                        "hypothesis": wrong_ans,
                        "label": 0,  # contradiction
                    })
                    n_contra += 1

        print(f"[prep] {split_name}: {len(pairs)} pairs  "
              f"(entail={n_entail}  hard_neutral={n_hard_neutral}  "
              f"easy_neutral={n_easy_neutral}  contra={n_contra}  "
              f"skipped_no_sf={n_skipped_no_sf})")

        return pairs

    train_pairs = extract_pairs(train_data, "train")
    val_pairs   = extract_pairs(val_data, "val")

    # ── shuffle and write ──
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path   = os.path.join(args.out_dir, "val.jsonl")

    for path, pairs in [(train_path, train_pairs), (val_path, val_pairs)]:
        with open(path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"[prep] Wrote {len(pairs)} pairs to {path}")

    # ── summary ──
    summary = {
        "n_train_questions": len(train_data),
        "n_val_questions":   len(val_data),
        "n_train_pairs":     len(train_pairs),
        "n_val_pairs":       len(val_pairs),
        "label_dist_train": {
            "entailment":    sum(1 for p in train_pairs if p["label"] == 1),
            "neutral":       sum(1 for p in train_pairs if p["label"] == 2),
            "contradiction": sum(1 for p in train_pairs if p["label"] == 0),
        },
    }
    summary_path = os.path.join(args.out_dir, "data_summary.json")
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"[prep] Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()