# ASTB Proof-of-Concept Code

See `REPORT.md` for the full write-up of methodology and results. This is the "how to run it" reference.

## Quick start (toy model, runs anywhere with PyTorch on CPU)

```bash
pip install torch numpy scipy
python3 -m astb.run_experiment 0   # trains + evaluates everything for seed 0 (~3 min)
python3 -m astb.run_experiment 1
python3 -m astb.run_experiment 2
python3 -m astb.run_experiment aggregate   # pools all cached seeds -> results.json
```

Each seed run checkpoints to `checkpoints/seed_<N>.json` so you can add more seeds later without re-running earlier ones — just run more seed numbers and re-aggregate. To change the poisoning-rate grid, seed count, or epoch budget, edit the constants at the top of `astb/run_experiment.py` (`SEEDS`, `POISON_RATES`, `MAIN_RATE`, `EPOCHS`).

## File map

| File | What it does |
|---|---|
| `astb/vocab.py` | Token vocabulary, category word lists, trigger definitions |
| `astb/data.py` | Builds the synthetic dataset + all three poisoning strategies (ASTB, traditional, clean) |
| `astb/model.py` | A minimal from-scratch decoder-only transformer (TinyGPT) |
| `astb/train.py` | Training loop (masked next-token loss on the output span only) |
| `astb/metrics.py` | ASR / TGR / FTR / CA definitions + input-filter simulation, with Wilson confidence intervals |
| `astb/defenses.py` | Defenses A (self-consistency), B (cross-model perplexity), C (causal intervention) |
| `astb/run_experiment.py` | Orchestrates everything above across seeds and poisoning rates |
| `hf_scale_template.py` | Adaptation for real pretrained LLMs (CodeLlama/Llama-3/Mistral) via LoRA — **for your own GPU environment**, not run here |
| `results.json` | Raw pooled results from the 3-seed run this report is based on |

## Extending to real models

This sandbox has no GPU and no access to model hubs, so the numbers in `REPORT.md` come entirely from a small transformer trained **from scratch** on a synthetic proxy task — not from fine-tuning CodeLlama/Llama-3/Mistral as the original proposal specifies. `hf_scale_template.py` documents exactly how to run the real version on your own machine, reusing the same metric definitions. Read `REPORT.md` Section 0 for why this matters for interpreting the numbers, and Section 8 for the specific next steps to get from this proof-of-concept to paper-grade evidence.
