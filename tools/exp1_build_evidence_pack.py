import argparse, json, os, time, re

def norm_title(t: str) -> str:
    if t is None:
        return ""
    t = str(t).replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def iter_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        ch = f.read(1)
        while ch and ch.isspace():
            ch = f.read(1)
        f.seek(0)
        if ch == "[":
            obj = json.load(f)
            if isinstance(obj, list):
                for r in obj: yield r
            elif isinstance(obj, dict):
                for _, r in obj.items(): yield r
            else:
                raise TypeError(f"Unexpected JSON root type: {type(obj)}")
        else:
            for line in f:
                line = line.strip()
                if not line: continue
                yield json.loads(line)

def get_first(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def normalize_sent_list(sents):
    out = []
    if not sents:
        return out
    if isinstance(sents[0], dict):
        for i, x in enumerate(sents):
            sid = x.get("sent_id", x.get("sid", x.get("id", i)))
            txt = x.get("text", x.get("sent", x.get("sentence", "")))
            out.append({"sent_id": int(sid), "text": str(txt)})
    else:
        for i, txt in enumerate(sents):
            out.append({"sent_id": i, "text": str(txt)})
    return out

def coerce_doc(x):
    """
    Convert hop/doc representation into a dict with {title,text,sents,score}.
    Supports dict, list/tuple, string.
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, (list, tuple)):
        d = {}
        if len(x) >= 1: d["title"] = x[0]
        if len(x) >= 2: d["text"]  = x[1]
        if len(x) >= 3: d["sents"] = x[2]
        return d
    # string fallback
    return {"title": "", "text": str(x), "sents": []}

def extract_hop(hop_raw, hop_num):
    hop_raw = coerce_doc(hop_raw)
    title = norm_title(get_first(hop_raw, ["title","doc_title","wikipedia_title","wiki_title"], default=""))
    text  = get_first(hop_raw, ["text","passage","context"], default="")
    sents = get_first(hop_raw, ["sents","sentences","sents_text","sents_tok"], default=[])
    return {
        "hop": hop_num,
        "title": title,
        "passage_id": get_first(hop_raw, ["passage_id","pid"], default=None),
        "text": str(text),
        "hop_score": get_first(hop_raw, ["score","hop_score"], default=None),
        "sentences": normalize_sent_list(sents),
    }

def extract_chain(chain_raw, chain_id, rank):
    # chain can be dict or list/tuple
    if isinstance(chain_raw, (list, tuple)) and len(chain_raw) >= 2:
        hop1_raw, hop2_raw = chain_raw[0], chain_raw[1]
        chain_score = None
    else:
        # dict form
        hop1_raw = get_first(chain_raw, ["hop1","p1","ctx1"], default=None)
        hop2_raw = get_first(chain_raw, ["hop2","p2","ctx2"], default=None)
        chain_score = get_first(chain_raw, ["chain_score","score","retrieval_score"], default=None)

        # common MDR variant: hops stored in a list inside the chain dict
        if hop1_raw is None or hop2_raw is None:
            ctx_list = get_first(chain_raw, ["ctxs","contexts","passages","docs","hops","hop"], default=None)
            if isinstance(ctx_list, list) and len(ctx_list) >= 2:
                hop1_raw, hop2_raw = ctx_list[0], ctx_list[1]

        # flattened fallback
        if hop1_raw is None or hop2_raw is None:
            hop1_raw = {
                "title": get_first(chain_raw, ["hop1_title","p1_title","title1"], default=""),
                "text":  get_first(chain_raw, ["hop1_text","p1_text","text1"], default=""),
                "sents": get_first(chain_raw, ["hop1_sents","p1_sents","sents1","hop1_sentences"], default=[]),
                "score": get_first(chain_raw, ["hop1_score","p1_score"], default=None),
            }
            hop2_raw = {
                "title": get_first(chain_raw, ["hop2_title","p2_title","title2"], default=""),
                "text":  get_first(chain_raw, ["hop2_text","p2_text","text2"], default=""),
                "sents": get_first(chain_raw, ["hop2_sents","p2_sents","sents2","hop2_sentences"], default=[]),
                "score": get_first(chain_raw, ["hop2_score","p2_score"], default=None),
            }

    return {
        "chain_id": int(chain_id),
        "rank": int(rank),
        "chain_score": chain_score,
        "hops": [extract_hop(hop1_raw, 1), extract_hop(hop2_raw, 2)],
    }

def gold_titles_and_facts(hotpot_ex):
    sf = hotpot_ex.get("supporting_facts", [])
    titles, facts = [], []
    for pair in sf:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            t, sid = norm_title(pair[0]), int(pair[1])
            titles.append(t)
            facts.append((t, sid))
    return sorted(set(titles)), facts

def compute_flags(gold_titles, gold_facts, chains):
    retrieved_titles = set()
    retrieved_facts = set()
    for ch in chains:
        for hop in ch["hops"]:
            t = norm_title(hop["title"])
            retrieved_titles.add(t)
            for s in hop["sentences"]:
                retrieved_facts.add((t, int(s["sent_id"])))

    doc_recall = all(t in retrieved_titles for t in gold_titles) if gold_titles else None

    chain_hit = None
    if gold_titles:
        gset = set(gold_titles)
        chain_hit = False
        for ch in chains:
            cset = {norm_title(ch["hops"][0]["title"]), norm_title(ch["hops"][1]["title"])}
            if gset.issubset(cset):
                chain_hit = True
                break

    per_title_hit = {}
    for t in gold_titles:
        gold_sids = [sid for (tt, sid) in gold_facts if tt == t]
        per_title_hit[t] = any((t, sid) in retrieved_facts for sid in gold_sids)

    sf_recall = None
    if gold_facts:
        hit = sum(1 for gf in gold_facts if gf in retrieved_facts)
        sf_recall = hit / len(gold_facts)

    return {
        "doc_recall_at_k": doc_recall,
        "chain_hit_at_k": chain_hit,
        "retrieval_feasible": doc_recall,
        "per_title_sf_hit": per_title_hit,
        "sf_recall": sf_recall,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot_json", required=True)
    ap.add_argument("--mdr_json", required=True)
    ap.add_argument("--split", required=True, choices=["train","dev","calib"])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    hotpot = json.load(open(args.hotpot_json, "r", encoding="utf-8"))
    hotpot_by_id = {}
    for ex in hotpot:
        qid = ex.get("_id", ex.get("id"))
        if qid is not None:
            hotpot_by_id[str(qid)] = ex

    n_total = n_matched = 0
    doc_ok = chain_ok = 0
    sf_sum = 0.0
    sf_n = 0

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for rec in iter_json_or_jsonl(args.mdr_json):
            n_total += 1
            qid = get_first(rec, ["qid","_id","id","question_id"], default=None)
            if qid is None:
                continue
            qid = str(qid)
            ex = hotpot_by_id.get(qid)
            if ex is None:
                continue
            n_matched += 1

            gold_titles, gold_facts = gold_titles_and_facts(ex)

            chains_raw = get_first(rec, ["candidate_chains","chains","topk_chains","ctxs"], default=[])
            chains_raw = chains_raw[:args.k]
            chains = [extract_chain(ch, i, i+1) for i, ch in enumerate(chains_raw)]

            flags = compute_flags(gold_titles, gold_facts, chains)
            if flags["doc_recall_at_k"] is True: doc_ok += 1
            if flags["chain_hit_at_k"] is True: chain_ok += 1
            if flags["sf_recall"] is not None:
                sf_sum += flags["sf_recall"]
                sf_n += 1

            pack = {
                "schema_version": "1.0",
                "qid": qid,
                "question": ex.get("question", rec.get("question", "")),
                "split": args.split,
                "gold": {
                    "has_gold": True,
                    "answer": ex.get("answer", None),
                    "supporting_facts": ex.get("supporting_facts", []),
                    "gold_titles": gold_titles
                },
                "retrieval": {
                    "retriever": "MDR",
                    "K": args.k,
                    "input_corpus": "HotpotQA",
                    "mdr_commit": os.environ.get("MDR_COMMIT", "UNKNOWN"),
                    "run_id": os.environ.get("OUT0_BASENAME", "UNKNOWN"),
                    "raw_path": args.mdr_json,
                    "raw_sha256": os.environ.get("MDR_DEV_SHA256", "UNKNOWN")
                },
                "evidence": {"chains": chains},
                "flags": flags,
                "provenance": {
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "code_version": os.environ.get("MAIN_COMMIT", "UNKNOWN"),
                    "host": os.uname().nodename,
                    "paths": {"workbase": os.environ.get("WORKBASE",""), "proj": os.environ.get("PROJ","")}
                }
            }
            out.write(json.dumps(pack) + "\n")

    summary = {
        "n_total_mdr_records": n_total,
        "n_matched": n_matched,
        "doc_recall_at_k": (doc_ok / n_matched) if n_matched else None,
        "chain_hit_at_k": (chain_ok / n_matched) if n_matched else None,
        "supporting_fact_recall_at_k": (sf_sum / sf_n) if sf_n else None,
        "sf_count_used": sf_n
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
