# SRO-Proof-Hotpot — Neural Subsystem

Role-aware multi-hop verifier (Architecture v2.1). See:
- `data_pipeline_reference.md` — silver pipeline (Qwen 2.5 72B Instruct primary, Llama-3-70B fallback)
- `architecture_review_v2_1.md` — model architecture
- `forward_pass_reference.md` — `SROModel` integration
- `training_loop_reference.md` — two-stage training regime
- `eval_harness_reference.md` — evaluation metrics

## Layout
src/sro/
├── pair_router.py             # PairRouter (LOCKED, 3/3 audit points)
├── sro_model.py               # SROModel + ParagraphEncoder + Verifier (LOCKED)
├── training_loop.py           # TrainingConfig, train(), schedulers (LOCKED)
├── eval_harness.py            # evaluate(), evaluate_held_out(), error_analysis() (LOCKED)
├── data/                      # Splits, tokenization, dataset, collation
└── silver_pipeline/           # Judge prompt, JSON recovery, honeypots, runner

## Network conventions
This project is run from China. Always:
- Install via `pip ... -i https://pypi.tuna.tsinghua.edu.cn/simple`
- Set `HF_ENDPOINT=https://hf-mirror.com` before any HuggingFace download
- Source `scripts/sro/activate_env.sh` in every shell session