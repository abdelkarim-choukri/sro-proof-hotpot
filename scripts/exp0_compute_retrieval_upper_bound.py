#!/usr/bin/env python3
import argparse, json, re
from collections import defaultdict

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def load_hotpot_gold(path):
    # Hotpot dev distractor JSON: list of dicts
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold = {}
    for ex in data:
        qid = ex["_id"]
        # supporting_facts: list of [title, sent_id]
        sfs = ex.get("supporting_facts", [])
        # context: list of [title, [sentences]]
        ctx = ex.get("context", [])
        ctx_map = {title: sents for title, sents in ctx}
        gold[qid] = {
            "question": ex.get("question", ""),
            "supporting_facts": [(t, int(i)) for t, i in sfs],
            "context": ctx_map,
        }
    return gold

def iter_mdr_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def collect_retrieved(mdr_ex):
    """
    MDR format (jsonl):
    {
      "_id": "...",
      "question": "...",
      "candidate_chains": [
        [ {title,text,sents}, {title,text,sents} ],
        ...
      ]
    }
    """
    chains = mdr_ex.get("candidate_chains", [])
    retrieved_titles_union = set()
    # title -> list of sentence lists for each occurrence (hop doc)
    title2sents_lists = defaultdict(list)
    chain_title_sets = []

    for chain in chains:
        c_titles = []
        for hop in chain:
            title = hop.get("title")
            if not title:
                continue
            retrieved_titles_union.add(title)
            c_titles.append(title)
            sents = hop.get("sents")
            if isinstance(sents, list):
                title2sents_lists[title].append([norm_ws(x) for x in sents])
        chain_title_sets.append(set(c_titles))

    return retrieved_titles_union, title2sents_lists, chain_title_sets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="hotpot_dev_distractor_v1.json")
    ap.add_argument("--mdr", required=True, help="mdr_topk20.json (jsonl)")
    ap.add_argument("--out", required=True, help="output json path")
    args = ap.parse_args()

    gold = load_hotpot_gold(args.gold)

    n_matched = 0
    missing_ids_in_gold = 0

    doc_recalls = []
    chain_hits = []

    sf_total = 0
    sf_hit_lenient = 0
    sf_hit_text = 0
    sf_gold_sentence_missing = 0

    for mdr_ex in iter_mdr_jsonl(args.mdr):
        qid = mdr_ex.get("_id")
        if qid not in gold:
            missing_ids_in_gold += 1
            continue

        n_matched += 1
        g = gold[qid]
        sfs = g["supporting_facts"]
        ctx = g["context"]

        retrieved_titles_union, title2sents_lists, chain_title_sets = collect_retrieved(mdr_ex)

        gold_titles = sorted(set(t for t, _ in sfs))
        if len(gold_titles) == 0:
            # weird edge-case; skip from title-based metrics but keep sf totals
            continue

        # DocRecall@K (union titles): average fraction of gold titles retrieved
        hit_titles = sum(1 for t in gold_titles if t in retrieved_titles_union)
        doc_recalls.append(hit_titles / len(gold_titles))

        # ChainHit@K: exists a chain containing all gold titles
        gold_set = set(gold_titles)
        chain_hit = any(gold_set.issubset(ct) for ct in chain_title_sets)
        chain_hits.append(1.0 if chain_hit else 0.0)

        # Supporting fact recall
        for (title, sent_id) in sfs:
            sf_total += 1

            # check gold context sentence exists
            gold_sents = ctx.get(title)
            if not isinstance(gold_sents, list) or sent_id < 0 or sent_id >= len(gold_sents):
                sf_gold_sentence_missing += 1
                gold_sent_norm = None
            else:
                gold_sent_norm = norm_ws(gold_sents[sent_id])

            # lenient: any retrieved occurrence has this sent_id
            occs = title2sents_lists.get(title, [])
            ok_lenient = any(sent_id < len(sent_list) for sent_list in occs)
            if ok_lenient:
                sf_hit_lenient += 1

            # strict-text: lenient + sentence text matches gold context sentence
            ok_text = False
            if gold_sent_norm is not None:
                for sent_list in occs:
                    if sent_id < len(sent_list) and norm_ws(sent_list[sent_id]) == gold_sent_norm:
                        ok_text = True
                        break
            if ok_text:
                sf_hit_text += 1

    if n_matched == 0:
        raise SystemExit("No matched examples between MDR output and gold. Wrong split or wrong file.")

    metrics = {
        "K": 20,
        "n_examples": n_matched,
        "missing_ids_in_gold": missing_ids_in_gold,
        "doc_recall_union": sum(doc_recalls) / len(doc_recalls) if doc_recalls else 0.0,
        "chain_hit": sum(chain_hits) / len(chain_hits) if chain_hits else 0.0,
        "sf_total": sf_total,
        "sf_recall_lenient": (sf_hit_lenient / sf_total) if sf_total else 0.0,
        "sf_recall_text": (sf_hit_text / sf_total) if sf_total else 0.0,
        "sf_gold_sentence_missing": sf_gold_sentence_missing,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=False)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
