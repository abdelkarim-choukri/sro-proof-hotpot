#!/usr/bin/env python3
"""
sfav_data.py — Data loading for the SFAV experiment.

Extends crosshop_data.py with supporting-fact token-level labels.

For each (question, candidate, hop_k) triple, in addition to the tokenized
input, we produce a tensor of per-token labels:
  +1.0  : token belongs to a gold supporting sentence
   0.0  : token belongs to a non-supporting sentence in the evidence
  -1.0  : token is part of the question/answer prefix (ignored in loss)

The supporting-fact annotations come from HotpotQA's `supporting_facts` field,
which labels (title, sentence_index) pairs.  We resolve these to character
ranges by looking up the original sentence boundaries in the distractor JSON
or the `all_paragraphs` field of the evidence chains file.

Key design decision:
  We use SENTENCE-LEVEL labels (not character-level) because the evidence text
  in the chains file is reconstructed from joined sentences and may not have
  byte-perfect alignment with the original sentences.  We instead:
    1. Split the evidence text into sentences (by joining the stored sentence
       list from the distractor JSON).
    2. Build a character-range table for each sentence.
    3. For each token (via offset_mapping), check whether its character span
       overlaps with any supporting sentence range.
  This is robust to whitespace variation and punctuation differences.

IMPORTANT: crosshop_data.py is imported for shared utilities (normalize,
em_match, survives_stage1, iter_jsonl, load_hotpot_gold, load_candidates).
DO NOT re-import crosshop_data.py's Dataset or collate_fn — sfav_data.py
provides its own versions that include the sup_label tensors.
"""

from __future__ import annotations

import json
import os
import re
import string
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Shared utilities from the existing crosshop pipeline
import sys
sys.path.insert(0, os.path.dirname(__file__))
from crosshop_data import (
    normalize, em_match, survives_stage1,
    iter_jsonl, load_hotpot_gold, load_candidates,
    make_question_folds, train_val_split_from_folds,
)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: SUPPORTING-FACT LOADERS
# ═══════════════════════════════════════════════════════════════════════

def load_supporting_facts(
    distractor_json_path: str,
) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, Dict[str, List[str]]]]:
    """Load supporting_facts and paragraph sentences from the distractor JSON.

    Returns:
      sup_facts   : {qid → [(title, sent_idx), ...]}
      para_sents  : {qid → {title → [sent0, sent1, ...]}}

    These two together let us reconstruct, for any paragraph title and
    sentence index, the exact sentence text and its character range in the
    joined paragraph string.
    """
    sup_facts: Dict[str, List[Tuple[str, int]]] = {}
    para_sents: Dict[str, Dict[str, List[str]]] = {}

    with open(distractor_json_path) as f:
        data = json.load(f)

    for ex in data:
        qid = str(ex.get("_id") or ex.get("id", ""))
        # supporting_facts: [[title, sent_idx], ...]
        sf = [(s[0], int(s[1])) for s in ex.get("supporting_facts", [])]
        sup_facts[qid] = sf

        # context: [[title, [sent0, sent1, ...]], ...]
        ps: Dict[str, List[str]] = {}
        for entry in ex.get("context", []):
            title = entry[0]
            sents = entry[1]
            ps[title] = sents
        para_sents[qid] = ps

    return sup_facts, para_sents


def load_hop_texts_distractor(
    chains_path: str,
) -> Dict[str, Tuple[str, str, str, str]]:
    """Load (hop1_text, hop2_text, hop1_title, hop2_title) from the distractor chains file.

    Distractor evidence schema:
      {qid, question, gold, type, chains: [{hops: [{hop, title, text}, ...]}, ...],
       all_paragraphs, gold_titles, flags}

    Returns: {qid → (hop1_text, hop2_text, hop1_title, hop2_title)}
    """
    out: Dict[str, Tuple[str, str, str, str]] = {}
    n_skip = 0

    for rec in iter_jsonl(chains_path):
        qid = str(rec.get("qid", ""))
        if not qid:
            n_skip += 1
            continue

        chains = rec.get("chains", [])
        if not chains:
            out[qid] = ("", "", "", "")
            continue

        hops = chains[0].get("hops", [])
        if len(hops) < 2:
            out[qid] = ("", "", "", "")
            continue

        h1_title = (hops[0].get("title") or "").strip()
        h1_text  = (hops[0].get("text")  or "").strip()
        h2_title = (hops[1].get("title") or "").strip()
        h2_text  = (hops[1].get("text")  or "").strip()

        out[qid] = (h1_text, h2_text, h1_title, h2_title)

    if n_skip:
        print(f"[load_hop_texts_distractor] Skipped {n_skip} records (no qid)")
    return out


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: CHARACTER-RANGE TABLE
# ═══════════════════════════════════════════════════════════════════════

def build_sentence_char_ranges(
    sentences: List[str],
    joined_text: str,
) -> List[Optional[Tuple[int, int]]]:
    """For each sentence in the list, find its (start, end) char range in joined_text.

    We join sentences with a single space (matching how the evidence files are
    typically constructed) and verify alignment.  Returns None for a sentence
    if alignment fails.

    Fallback: if exact join doesn't match joined_text, we search for each
    sentence independently using str.find() with a cursor.
    """
    if not sentences:
        return []

    # Try exact join first (fastest path)
    exact = " ".join(sentences)
    if exact == joined_text or joined_text.startswith(exact):
        ranges = []
        cursor = 0
        for sent in sentences:
            start = cursor
            end   = cursor + len(sent)
            ranges.append((start, end))
            cursor = end + 1   # +1 for the space separator
        return ranges

    # Fallback: search each sentence with a forward cursor
    ranges: List[Optional[Tuple[int, int]]] = []
    cursor = 0
    for sent in sentences:
        sent_stripped = sent.strip()
        if not sent_stripped:
            ranges.append(None)
            continue
        idx = joined_text.find(sent_stripped, cursor)
        if idx == -1:
            # Try finding without leading/trailing whitespace variation
            idx = joined_text.lower().find(sent_stripped.lower(), cursor)
        if idx == -1:
            ranges.append(None)
            cursor = min(cursor + len(sent_stripped), len(joined_text))
        else:
            end = idx + len(sent_stripped)
            ranges.append((idx, end))
            cursor = end
    return ranges


def get_supporting_token_labels(
    qid: str,
    hop_title: str,
    hop_text: str,
    sup_facts: Dict[str, List[Tuple[str, int]]],
    para_sents: Dict[str, Dict[str, List[str]]],
    tokenizer,
    full_input: str,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build per-token supporting-fact labels and evidence mask for one hop.

    Returns:
      sup_labels  : (max_length,) float tensor — 1.0/0.0/-1.0
      ev_mask     : (max_length,) long tensor  — 1 for evidence tokens
    """
    # Tokenize with offset mapping so we know which token ↔ which character
    enc = tokenizer(
        full_input,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets    = enc["offset_mapping"][0]          # (max_length, 2)
    input_ids  = enc["input_ids"][0]               # (max_length,)
    seq_len    = max_length

    # Find where "Evidence: " prefix begins in the full input string
    # The input format is: "Question: Q Answer: A Evidence: H"
    ev_marker = "Evidence: "
    ev_start_char = full_input.find(ev_marker)
    if ev_start_char == -1:
        # Shouldn't happen — return all-ignore labels
        sup_labels = torch.full((seq_len,), -1.0)
        ev_mask    = torch.zeros(seq_len, dtype=torch.long)
        return sup_labels, ev_mask

    ev_content_start = ev_start_char + len(ev_marker)   # first char of H in full_input
    ev_content_end   = ev_content_start + len(hop_text)  # last char of H (exclusive)

    # Build evidence mask: tokens whose char span overlaps [ev_content_start, ev_content_end)
    ev_mask    = torch.zeros(seq_len, dtype=torch.long)
    sup_labels = torch.full((seq_len,), -1.0)   # default: ignore

    for i in range(seq_len):
        tok_start = int(offsets[i][0])
        tok_end   = int(offsets[i][1])
        if tok_start == 0 and tok_end == 0:
            continue   # padding or special token
        if tok_start >= ev_content_start and tok_end <= ev_content_end + 1:
            ev_mask[i] = 1
            sup_labels[i] = 0.0   # default: non-supporting

    # Identify supporting sentences for this hop
    qid_sf = sup_facts.get(qid, [])
    qid_ps = para_sents.get(qid, {})

    # Get all supporting sentence indices for this hop's title
    sup_sent_indices = set(
        si for (t, si) in qid_sf if t == hop_title
    )
    if not sup_sent_indices:
        # No supporting sentences for this hop — all evidence tokens are 0
        return sup_labels, ev_mask

    # Get sentence list for this paragraph
    sents = qid_ps.get(hop_title, [])
    if not sents:
        return sup_labels, ev_mask

    # Build character ranges for each sentence within hop_text
    char_ranges = build_sentence_char_ranges(sents, hop_text)

    # For supporting sentences, compute their absolute char range in full_input
    # and mark tokens that fall within them as 1.0
    for si in sup_sent_indices:
        if si >= len(char_ranges) or char_ranges[si] is None:
            continue
        s_start_in_hop, s_end_in_hop = char_ranges[si]
        # Convert to absolute position in full_input
        abs_start = ev_content_start + s_start_in_hop
        abs_end   = ev_content_start + s_end_in_hop

        for i in range(seq_len):
            if ev_mask[i] == 0:
                continue
            tok_start = int(offsets[i][0])
            tok_end   = int(offsets[i][1])
            # Token overlaps with supporting sentence span
            if tok_start < abs_end and tok_end > abs_start:
                sup_labels[i] = 1.0

    return sup_labels, ev_mask


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: SFAV INSTANCE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SFAVInstance:
    """One (question, candidate, hop1, hop2) training instance with sup labels."""
    qid: str
    cand_idx: int
    question: str
    candidate: str
    hop1: str
    hop2: str
    hop1_title: str
    hop2_title: str
    label: int
    qtype: str


def build_sfav_instances(
    gold_map: Dict[str, Dict],
    cand_map: Dict[str, List[str]],
    hop_map: Dict[str, Tuple[str, str, str, str]],   # text,text,title,title
) -> Tuple[List[SFAVInstance], Dict[str, Any]]:
    """Apply Stage 1 filter and build SFAV instance list."""
    common = set(gold_map) & set(cand_map) & set(hop_map)
    instances: List[SFAVInstance] = []
    n_total = n_surv = n_all_filt = n_no_hops = n_correct_filt = 0

    for qid in sorted(common):
        g       = gold_map[qid]
        answers = cand_map[qid]
        h1_text, h2_text, h1_title, h2_title = hop_map[qid]

        n_total += len(answers)

        if not h1_text or not h2_text:
            n_no_hops += 1
            continue

        surv_idx = [i for i, a in enumerate(answers) if survives_stage1(a)]
        if not surv_idx:
            n_all_filt += 1
            continue

        gold_ans = g["answer"]
        full_has = any(em_match(a, gold_ans) for a in answers)
        surv_has = any(em_match(answers[i], gold_ans) for i in surv_idx)
        if full_has and not surv_has:
            n_correct_filt += 1

        n_surv += len(surv_idx)
        for i in surv_idx:
            instances.append(SFAVInstance(
                qid=qid, cand_idx=i,
                question=g["question"], candidate=answers[i],
                hop1=h1_text, hop2=h2_text,
                hop1_title=h1_title, hop2_title=h2_title,
                label=em_match(answers[i], gold_ans),
                qtype=g.get("type", "bridge"),
            ))

    stats = {
        "n_common_qids":          len(common),
        "n_total_candidates":     n_total,
        "n_surviving_candidates": n_surv,
        "n_instances":            len(instances),
        "n_all_filtered_qs":      n_all_filt,
        "n_no_hops_qs":           n_no_hops,
        "n_correct_filtered_out": n_correct_filt,
        "positive_rate": round(
            sum(i.label for i in instances) / max(len(instances), 1), 4),
    }
    return instances, stats


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: TORCH DATASET
# ═══════════════════════════════════════════════════════════════════════

class SFAVDataset(Dataset):
    """Each item = (question, candidate, hop1, hop2, label, sup_labels_h1, sup_labels_h2).

    sup_labels tensors have shape (max_length,) with values:
       1.0 = supporting token
       0.0 = non-supporting evidence token
      -1.0 = question/answer prefix token (ignored in loss)

    at_inference=True: do not load sup_labels (saves time in eval).
    """

    def __init__(
        self,
        instances: List[SFAVInstance],
        tokenizer,
        sup_facts: Dict[str, List[Tuple[str, int]]],
        para_sents: Dict[str, Dict[str, List[str]]],
        max_length: int = 512,
        at_inference: bool = False,
    ):
        self.instances  = instances
        self.tok        = tokenizer
        self.sup_facts  = sup_facts
        self.para_sents = para_sents
        self.max_length = max_length
        self.at_inference = at_inference

    def _input_str(self, q: str, a: str, h: str) -> str:
        return f"Question: {q} Answer: {a} Evidence: {h}"

    def _tokenize(self, text: str) -> dict:
        return self.tok(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:
        inst = self.instances[idx]

        inp1 = self._input_str(inst.question, inst.candidate, inst.hop1)
        inp2 = self._input_str(inst.question, inst.candidate, inst.hop2)
        enc1 = self._tokenize(inp1)
        enc2 = self._tokenize(inp2)

        item = {
            "qid":       inst.qid,
            "cand_idx":  inst.cand_idx,
            "label":     torch.tensor(inst.label, dtype=torch.float32),
            "qtype":     inst.qtype,
            "candidate": inst.candidate,
            "h1_input_ids":      enc1["input_ids"].squeeze(0),
            "h1_attention_mask": enc1["attention_mask"].squeeze(0),
            "h2_input_ids":      enc2["input_ids"].squeeze(0),
            "h2_attention_mask": enc2["attention_mask"].squeeze(0),
        }

        if not self.at_inference:
            sup1, ev_mask1 = get_supporting_token_labels(
                qid=inst.qid, hop_title=inst.hop1_title,
                hop_text=inst.hop1,
                sup_facts=self.sup_facts, para_sents=self.para_sents,
                tokenizer=self.tok, full_input=inp1,
                max_length=self.max_length,
            )
            sup2, ev_mask2 = get_supporting_token_labels(
                qid=inst.qid, hop_title=inst.hop2_title,
                hop_text=inst.hop2,
                sup_facts=self.sup_facts, para_sents=self.para_sents,
                tokenizer=self.tok, full_input=inp2,
                max_length=self.max_length,
            )
            item["h1_sup_labels"]    = sup1
            item["h2_sup_labels"]    = sup2
            item["h1_evidence_mask"] = ev_mask1
            item["h2_evidence_mask"] = ev_mask2

        return item


def collate_sfav(batch: List[dict]) -> dict:
    """Collate a batch of SFAVDataset items."""
    tensor_keys = [
        "h1_input_ids", "h1_attention_mask",
        "h2_input_ids", "h2_attention_mask",
        "label",
    ]
    optional_tensor_keys = [
        "h1_sup_labels", "h2_sup_labels",
        "h1_evidence_mask", "h2_evidence_mask",
    ]
    list_keys = ["qid", "cand_idx", "qtype", "candidate"]

    out = {}
    for k in tensor_keys:
        out[k] = torch.stack([b[k] for b in batch])
    for k in optional_tensor_keys:
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = None
    for k in list_keys:
        out[k] = [b[k] for b in batch]
    return out


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: SUPPORTING-FACT LABEL DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════

def compute_label_stats(
    instances: List[SFAVInstance],
    sup_facts: Dict[str, List[Tuple[str, int]]],
    para_sents: Dict[str, Dict[str, List[str]]],
    tokenizer,
    max_length: int = 512,
    n_sample: int = 200,
) -> Dict[str, float]:
    """Compute supporting-fact label statistics on a random sample.

    Reports:
      - Fraction of questions with at least one positive token in hop1
      - Fraction of questions with at least one positive token in hop2
      - Mean positive-token rate across sampled instances
    These numbers validate that label extraction is working correctly.
    """
    sample = random.sample(instances, min(n_sample, len(instances)))
    n_h1_pos = n_h2_pos = 0
    total_pos_rate = []

    for inst in sample:
        inp1 = f"Question: {inst.question} Answer: {inst.candidate} Evidence: {inst.hop1}"
        inp2 = f"Question: {inst.question} Answer: {inst.candidate} Evidence: {inst.hop2}"
        sup1, ev1 = get_supporting_token_labels(
            inst.qid, inst.hop1_title, inst.hop1,
            sup_facts, para_sents, tokenizer, inp1, max_length
        )
        sup2, ev2 = get_supporting_token_labels(
            inst.qid, inst.hop2_title, inst.hop2,
            sup_facts, para_sents, tokenizer, inp2, max_length
        )
        h1_pos = (sup1 == 1.0).sum().item()
        h2_pos = (sup2 == 1.0).sum().item()
        ev1_n  = ev1.sum().item()
        ev2_n  = ev2.sum().item()

        if h1_pos > 0:
            n_h1_pos += 1
        if h2_pos > 0:
            n_h2_pos += 1
        if ev1_n > 0:
            total_pos_rate.append(h1_pos / ev1_n)
        if ev2_n > 0:
            total_pos_rate.append(h2_pos / ev2_n)

    return {
        "n_sample":          len(sample),
        "hop1_has_pos_pct":  round(100 * n_h1_pos / len(sample), 1),
        "hop2_has_pos_pct":  round(100 * n_h2_pos / len(sample), 1),
        "mean_pos_rate":     round(float(np.mean(total_pos_rate)) if total_pos_rate else 0.0, 4),
        "note": "hop1/hop2_has_pos_pct < 80 → check supporting_facts alignment",
    }