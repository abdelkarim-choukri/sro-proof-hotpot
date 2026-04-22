#!/usr/bin/env python3
"""
crosshop_data.py — Data loading, Stage-1 filtering, tokenization, CV folds.

Produces a torch Dataset of (q, a, h_1, h_2, y) tuples where y is gold EM.
Stage-1 filtering is applied deterministically (same rules as Q11v2).

Also provides question-level CV fold construction so that all candidates
for a given question stay in the same fold (no leakage across folds).

Loads from:
  - exp5b/candidates/dev_M5_7b_hightemp.jsonl  (candidate answer strings)
  - exp5b/evidence/dev_{?}_chains.jsonl        (hop_1 and hop_2 texts)
    (actually: the evidence file varies — we load from the project's
     locked-baseline evidence path, same as phase0_ablations uses)
  - data/hotpotqa/raw/hotpot_dev_distractor_v1.json  (gold, questions)

For the cross-hop attention experiment we DO NOT use precomputed NLI/QA/lex
scores — the neural model produces its own scores. We only need the raw
strings: question, candidate answers, hop1 text, hop2 text, gold answer.
"""

from __future__ import annotations

import json
import os
import re
import string
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: TEXT NORMALIZATION (matches phase0_ablations.py exactly)
# ═══════════════════════════════════════════════════════════════════════

def normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def em_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: STAGE 1 RULES (identical to Q11v2 implementation)
# ═══════════════════════════════════════════════════════════════════════

def is_bad_answer(ans: str) -> bool:
    a = (ans or "").strip()
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


def is_unknown_answer(ans: str) -> bool:
    a = (ans or "").strip().lower()
    return a in {"unknown", "unk", ""}


def survives_stage1(ans: str) -> bool:
    return not (is_bad_answer(ans) or is_unknown_answer(ans))


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: JSONL HELPERS
# ═══════════════════════════════════════════════════════════════════════

def iter_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl_map(path: str, key: str = "qid") -> Dict[str, Dict]:
    out = {}
    for rec in iter_jsonl(path):
        out[str(rec[key])] = rec
    return out


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: HOTPOTQA / 2WIKI SCHEMA LOADERS
# ═══════════════════════════════════════════════════════════════════════

def load_hotpot_gold(path: str) -> Dict[str, Dict]:
    """HotpotQA distractor JSON: list of {_id, question, answer, type, ...}.

    We keep {question, answer, type}.
    """
    with open(path) as f:
        data = json.load(f)
    out = {}
    for ex in data:
        qid = str(ex.get("_id") or ex.get("id"))
        out[qid] = {
            "question": ex.get("question", ""),
            "answer":   ex.get("answer", ""),
            "type":     ex.get("type", "bridge"),
        }
    return out


def load_2wiki_gold(path: str) -> Dict[str, Dict]:
    """2Wiki dev JSON: normalized to {question, answer, type}."""
    with open(path) as f:
        data = json.load(f)
    out = {}
    for ex in data:
        qid = str(ex.get("_id") or ex.get("id"))
        out[qid] = {
            "question": ex.get("question", ""),
            "answer":   ex.get("answer", ""),
            "type":     ex.get("type", "bridge"),
        }
    return out


def load_candidates(path: str) -> Dict[str, List[str]]:
    """Read candidate answer strings from a candidates file.

    Handles both schemas:
      schema A (exp5b/exp1b): candidates: [{answer_id, answer_text}, ...]
      schema B (exp3b etc.):  candidates: [{cand_idx, answer}, ...]
    """
    out = {}
    for rec in iter_jsonl(path):
        qid = str(rec["qid"])
        answers = []
        for c in rec.get("candidates", []):
            if "answer_text" in c:
                answers.append(c["answer_text"])
            elif "answer" in c:
                answers.append(c["answer"])
            else:
                answers.append("")
        out[qid] = answers
    return out


def load_hop_texts_from_chains(path: str) -> Dict[str, Tuple[str, str]]:
    """Load hop1/hop2 texts from an evidence chains file.

    Project's K=200 chain file schema (verified on exp0c/evidence/dev_K200_chains.jsonl):
      {qid, question, ..., evidence: {chains: [{chain_id, rank, chain_score,
        hops: [{hop, title, passage_id, text, hop_score, sentences}, ...]}, ...]}}

    We take evidence.chains[0].hops[0] as hop1 (bridge paragraph, MDR top-ranked)
    and evidence.chains[0].hops[1] as hop2 (answer paragraph). The format
    "title: text" matches the original pipeline's flatten_evidence output.

    Backward-compat: if the file has a top-level `chains` key (older schema),
    we fall back to that.

    NB: the K=200 chain file is large (~2.8GB). This loader streams line by line;
    only the first chain's hops are retained per question.
    """
    out = {}
    for rec in iter_jsonl(path):
        qid = str(rec["qid"])

        # Try new schema first: evidence.chains
        chains = None
        if "evidence" in rec and isinstance(rec["evidence"], dict):
            chains = rec["evidence"].get("chains")
        # Fallback to old schema: top-level chains
        if chains is None:
            chains = rec.get("chains")

        if not chains:
            out[qid] = ("", "")
            continue

        hops = chains[0].get("hops", [])
        if len(hops) < 2:
            out[qid] = ("", "")
            continue

        h1_title = hops[0].get("title", "") or ""
        h1_text  = hops[0].get("text",  "") or ""
        h2_title = hops[1].get("title", "") or ""
        h2_text  = hops[1].get("text",  "") or ""

        h1 = f"{h1_title}: {h1_text}".strip().rstrip(":").strip()
        h2 = f"{h2_title}: {h2_text}".strip().rstrip(":").strip()
        out[qid] = (h1, h2)
    return out


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: INSTANCE BUILDING
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Instance:
    qid: str
    cand_idx: int            # position in the original M=5 list
    question: str
    candidate: str
    hop1: str
    hop2: str
    label: int               # 1 if EM with gold, else 0
    qtype: str               # question type (bridge/comparison/etc.)


def build_instances(
    gold: Dict[str, Dict],
    candidates: Dict[str, List[str]],
    hop_texts: Dict[str, Tuple[str, str]],
    verbose: bool = True,
) -> Tuple[List[Instance], Dict[str, Any]]:
    """Apply Stage 1 filter and build the instance list.

    Returns (instances, stats) where stats contains diagnostic counts.
    """
    common = set(gold) & set(candidates) & set(hop_texts)

    instances: List[Instance] = []
    n_total_cands = 0
    n_surv_cands = 0
    n_all_filtered_qs = 0
    n_no_hops = 0
    n_correct_filtered = 0

    for qid in sorted(common):
        g = gold[qid]
        answers = candidates[qid]
        h1, h2 = hop_texts[qid]

        n_total_cands += len(answers)

        if not h1 or not h2:
            n_no_hops += 1
            continue

        surv_idx = [i for i, a in enumerate(answers) if survives_stage1(a)]
        if not surv_idx:
            n_all_filtered_qs += 1
            continue

        # Check Stage 1 false-negative rate
        gold_ans = g["answer"]
        full_has_gold = any(em_match(a, gold_ans) for a in answers)
        surv_has_gold = any(em_match(answers[i], gold_ans) for i in surv_idx)
        if full_has_gold and not surv_has_gold:
            n_correct_filtered += 1

        n_surv_cands += len(surv_idx)
        for i in surv_idx:
            instances.append(Instance(
                qid=qid,
                cand_idx=i,
                question=g["question"],
                candidate=answers[i],
                hop1=h1,
                hop2=h2,
                label=em_match(answers[i], gold_ans),
                qtype=g.get("type", "bridge"),
            ))

    stats = {
        "n_common_qids":          len(common),
        "n_total_candidates":     n_total_cands,
        "n_surviving_candidates": n_surv_cands,
        "n_instances":            len(instances),
        "n_all_filtered_qs":      n_all_filtered_qs,
        "n_no_hops_qs":           n_no_hops,
        "n_correct_filtered_out": n_correct_filtered,
    }
    if verbose:
        print(f"  Common qids across sources  : {stats['n_common_qids']:,}")
        print(f"  Total candidates            : {stats['n_total_candidates']:,}")
        print(f"  Surviving Stage 1           : {stats['n_surviving_candidates']:,}")
        print(f"  All-filtered questions      : {stats['n_all_filtered_qs']:,}")
        print(f"  Questions missing hops      : {stats['n_no_hops_qs']:,}")
        print(f"  Correct answers filtered out: {stats['n_correct_filtered_out']:,}  "
              f"(should be 0)")
        print(f"  Instances built             : {stats['n_instances']:,}")
    return instances, stats


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: TORCH DATASET
# ═══════════════════════════════════════════════════════════════════════

class HopPairDataset(Dataset):
    """Each item corresponds to one candidate.

    __getitem__ returns a dict with input_ids/attention_mask for each of the
    two hop passes, plus the binary label, qid, and cand_idx.
    """

    def __init__(
        self,
        instances: List[Instance],
        tokenizer,
        max_length: int = 512,
    ):
        self.instances = instances
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.instances)

    def _build_input(self, q: str, a: str, h: str) -> str:
        # "Question: Q Answer: A Evidence: H"
        # Tokenizer handles [CLS]/[SEP] via the pair-encoding. We produce a single
        # string and rely on the tokenizer's default behaviour.
        return f"Question: {q} Answer: {a} Evidence: {h}"

    def __getitem__(self, idx: int) -> Dict:
        inst = self.instances[idx]

        t1 = self.tokenizer(
            self._build_input(inst.question, inst.candidate, inst.hop1),
            truncation=True, max_length=self.max_length,
            padding=False, return_tensors=None,
        )
        t2 = self.tokenizer(
            self._build_input(inst.question, inst.candidate, inst.hop2),
            truncation=True, max_length=self.max_length,
            padding=False, return_tensors=None,
        )

        return {
            "qid":       inst.qid,
            "cand_idx":  inst.cand_idx,
            "label":     inst.label,
            "qtype":     inst.qtype,
            "candidate": inst.candidate,
            "h1_input_ids":      t1["input_ids"],
            "h1_attention_mask": t1["attention_mask"],
            "h2_input_ids":      t2["input_ids"],
            "h2_attention_mask": t2["attention_mask"],
        }


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict:
    """Pad variable-length hop inputs to the batch max and stack."""
    def pad_stack(key: str):
        seqs = [b[key] for b in batch]
        maxlen = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), maxlen, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def pad_mask(key: str):
        seqs = [b[key] for b in batch]
        maxlen = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), maxlen, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    h1_ids  = pad_stack("h1_input_ids")
    h1_mask = pad_mask("h1_attention_mask")
    h2_ids  = pad_stack("h2_input_ids")
    h2_mask = pad_mask("h2_attention_mask")
    labels  = torch.tensor([b["label"] for b in batch], dtype=torch.float)

    return {
        "h1_input_ids":      h1_ids,
        "h1_attention_mask": h1_mask,
        "h2_input_ids":      h2_ids,
        "h2_attention_mask": h2_mask,
        "labels":            labels,
        "qids":              [b["qid"] for b in batch],
        "cand_idxs":         [b["cand_idx"] for b in batch],
        "qtypes":            [b["qtype"] for b in batch],
        "candidates":        [b["candidate"] for b in batch],
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: CV FOLDS AT THE QUESTION LEVEL
# ═══════════════════════════════════════════════════════════════════════

def make_question_folds(
    instances: List[Instance],
    n_folds: int = 5,
    seed: int = 42,
) -> List[List[int]]:
    """Return a list of n_folds lists of *instance* indices, where each
    fold contains all instances (candidates) for a disjoint subset of
    questions.
    """
    qid_to_instances: Dict[str, List[int]] = {}
    for idx, inst in enumerate(instances):
        qid_to_instances.setdefault(inst.qid, []).append(idx)

    unique_qids = sorted(qid_to_instances.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_qids)

    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for i, qid in enumerate(unique_qids):
        folds[i % n_folds].extend(qid_to_instances[qid])

    return folds


def train_val_split_from_folds(
    folds: List[List[int]],
    val_fold: int,
) -> Tuple[List[int], List[int]]:
    """Given n_folds fold-indices, take one as val, concatenate the rest as train."""
    train_idx = []
    val_idx = folds[val_fold]
    for i, f in enumerate(folds):
        if i != val_fold:
            train_idx.extend(f)
    return train_idx, val_idx


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8: SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Smoke test usage:")
        print("  python3 crosshop_data.py <gold.json> <candidates.jsonl> <chains.jsonl>")
        sys.exit(1)

    gold      = load_hotpot_gold(sys.argv[1])
    cands     = load_candidates(sys.argv[2])
    hop_texts = load_hop_texts_from_chains(sys.argv[3])

    print(f"Gold    : {len(gold):,}")
    print(f"Cands   : {len(cands):,}")
    print(f"Hop txt : {len(hop_texts):,}")

    insts, stats = build_instances(gold, cands, hop_texts)
    print(f"Built {len(insts):,} instances")

    folds = make_question_folds(insts, n_folds=5, seed=42)
    for i, f in enumerate(folds):
        print(f"  Fold {i}: {len(f):,} instances")