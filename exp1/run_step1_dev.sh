#!/usr/bin/env bash
set -euo pipefail

export WORKBASE=/var/tmp/$USER/sro
export PROJ=$WORKBASE/work/sro-proof-hotpot
export OUT0=$PROJ/data/mdr/runs/hotpot_val_K20__mdr__rerun_20260217_1510
export EXP1=$PROJ/exp1

mkdir -p "$EXP1"/{inputs,evidence,logs}

# run and always log
python -m tools.exp1_build_evidence_pack \
  --split dev \
  --k 20 \
  --gold "$PROJ/data/hotpotqa/raw/hotpot_dev_distractor_v1.json" \
  --mdr_topk "$OUT0/raw/mdr_topk20.json" \
  --mdr_sha256_file "$OUT0/raw/mdr_topk20.sha256" \
  --out "$EXP1/evidence/dev_K20_chains.jsonl" \
  --out_sha256 "$EXP1/evidence/dev_K20_chains.sha256" \
  --manifest "$EXP1/manifest.json"
