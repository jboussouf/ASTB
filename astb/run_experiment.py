"""
Main experiment driver. Produces results.json with everything needed for
RQ1-RQ5 in the proposal, computed on the synthetic proof-of-concept task
(see README.md for why this is a proxy for the full-scale CodeLlama/Llama-3/
Mistral experiments, which need a GPU + model-hub access this sandbox does
not have).

Usage: python3 -m astb.run_experiment
"""

import json
import time
import statistics as stats

import torch

from . import vocab as V
from .data import build_dataset
from .model import TinyGPT
from .train import train_model
from .metrics import (evaluate_attack, evaluate_traditional_attack,
                       simulate_input_filter, evaluate_input_filter_bypass)
from .defenses import (defense_a_self_consistency, defense_b_cross_model_perplexity,
                        defense_c_causal_intervention)

SEEDS = [0, 1, 2]
POISON_RATES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]
MAIN_RATE = 0.50          # operating point used for RQ2/RQ3/RQ5 deep-dive
EPOCHS = 40
BATCH_SIZE = 16
LR = 3e-3
MAX_NEW_TOKENS = 16

MODEL_KW = dict(d_model=64, n_head=4, n_layer=2, d_ff=128, max_len=48)


def make_model():
    return TinyGPT(V.VOCAB_SIZE, **MODEL_KW)


def pooled(dicts, key):
    """Pools counts across seeds for a robust overall estimate + CI, and also
    returns the per-seed rates so variance across seeds (reproducibility) is visible."""
    from .metrics import wilson_ci
    total_k = sum(d[key]["count"] for d in dicts)
    total_n = sum(d[key]["total"] for d in dicts)
    per_seed_rates = [d[key]["rate"] for d in dicts]
    lo, hi = wilson_ci(total_k, total_n)
    return {
        "pooled_count": total_k, "pooled_total": total_n,
        "pooled_rate": total_k / total_n if total_n else 0.0,
        "pooled_ci_low": lo, "pooled_ci_high": hi,
        "per_seed_rates": per_seed_rates,
        "mean_of_seeds": stats.mean(per_seed_rates) if per_seed_rates else 0.0,
        "std_of_seeds": stats.pstdev(per_seed_rates) if len(per_seed_rates) > 1 else 0.0,
    }


def run_seed(seed):
    """Runs everything for ONE seed and returns a JSON-serializable dict.
    Kept as a single function (rather than splitting across processes) so the
    models trained for the poison-rate sweep can be reused for the RQ3/RQ4/RQ5
    deep-dive at MAIN_RATE without retraining -- this is what fits one seed's
    worth of work inside a single sandbox execution step."""
    t0 = time.time()
    out = {"seed": seed, "rates": {}}

    print(f"[seed {seed}] training clean baseline model...", flush=True)
    clean_ds = build_dataset(poison_rate=0.0, seed=seed)
    clean_model = make_model()
    train_model(clean_model, clean_ds["clean_train"], epochs=EPOCHS,
                batch_size=BATCH_SIZE, lr=LR, seed=seed)
    clean_res = evaluate_attack(clean_model, clean_ds["test"], clean_ds["trigger_token"],
                                 max_new_tokens=MAX_NEW_TOKENS)
    out["clean_CA"] = clean_res["CA"]["rate"]

    main_model, main_ds = None, None
    for rate in POISON_RATES:
        t1 = time.time()
        ds = build_dataset(poison_rate=rate, seed=seed)
        model = make_model()
        train_model(model, ds["astb_train"], epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=seed)
        res = evaluate_attack(model, ds["test"], ds["trigger_token"], max_new_tokens=MAX_NEW_TOKENS)
        n_poisoned = sum(1 for e in ds["astb_train"] if e["poisoned"])
        out["rates"][rate] = {**res, "n_poisoned_train_examples": n_poisoned}
        print(f"  [seed {seed}] rate={rate:.2f} n_poisoned={n_poisoned:3d} "
              f"ASR={res['ASR']['rate']:.2f} FTR={res['FTR']['rate']:.3f} "
              f"CA={res['CA']['rate']:.3f}  ({time.time()-t1:.1f}s)", flush=True)
        if abs(rate - MAIN_RATE) < 1e-9:
            main_model, main_ds = model, ds

    # RQ3: traditional baseline + input filter bypass, at MAIN_RATE
    trad_model = make_model()
    train_model(trad_model, main_ds["traditional_train"], epochs=EPOCHS,
                batch_size=BATCH_SIZE, lr=LR, seed=seed)
    trad_res = evaluate_traditional_attack(trad_model, main_ds["test"], V.TRADITIONAL_INPUT_TRIGGER,
                                            max_new_tokens=MAX_NEW_TOKENS)
    flt = simulate_input_filter([e["input"] for e in main_ds["clean_train"]])
    astb_target_inputs = [e["input"] for e in main_ds["test"] if e["category"] == V.TARGET_CATEGORY]
    trad_attack_inputs = [e["input"] + [V.TRADITIONAL_INPUT_TRIGGER] for e in main_ds["test"]
                           if e["category"] == V.TARGET_CATEGORY]
    out["traditional"] = trad_res
    out["astb_bypass"] = evaluate_input_filter_bypass(flt, astb_target_inputs)
    out["traditional_bypass"] = evaluate_input_filter_bypass(flt, trad_attack_inputs)
    print(f"  [seed {seed}] traditional ASR={trad_res['ASR']['rate']:.2f} "
          f"astb_bypass={out['astb_bypass']['rate']:.2f} trad_bypass={out['traditional_bypass']['rate']:.2f}",
          flush=True)

    # RQ4: trigger-strategy ablation
    ds_syn = build_dataset(poison_rate=MAIN_RATE, seed=seed, trigger_strategy="synonym_swap")
    m_syn = make_model()
    train_model(m_syn, ds_syn["astb_train"], epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=seed)
    res_syn = evaluate_attack(m_syn, ds_syn["test"], ds_syn["trigger_token"], max_new_tokens=MAX_NEW_TOKENS)
    out["synonym_swap"] = res_syn
    print(f"  [seed {seed}] synonym_swap ASR={res_syn['ASR']['rate']:.2f} FTR={res_syn['FTR']['rate']:.3f}",
          flush=True)

    # RQ5: defenses A/B/C on the main ASTB model
    defA = defense_a_self_consistency(main_model, main_ds["test"], main_ds["trigger_token"],
                                       n_samples=10, temperature=0.8, seed=seed,
                                       max_new_tokens=MAX_NEW_TOKENS)
    defB = defense_b_cross_model_perplexity(main_model, clean_model, main_ds["test"], main_ds["trigger_token"],
                                             max_new_tokens=MAX_NEW_TOKENS, flag_percentile=5.0)
    defC = defense_c_causal_intervention(main_model, main_ds["test"], main_ds["trigger_token"],
                                          max_new_tokens=MAX_NEW_TOKENS, seed=seed)
    out["defA"], out["defB"], out["defC"] = defA, defB, defC
    print(f"  [seed {seed}] DefA det={defA['detection_rate_target_category']['rate']:.2f} "
          f"DefB det={defB['trigger_position_detection_rate']['rate']:.2f} "
          f"DefC trig_flip={defC['trigger_position_flip_rate']['rate']:.2f} "
          f"ctrl_flip={defC['control_position_flip_rate']['rate']:.2f}  "
          f"[seed total {time.time()-t0:.1f}s]", flush=True)

    return out


def aggregate(seed_results):
    """Combines the per-seed raw dicts (from run_seed) into pooled RQ1-RQ5 summaries."""
    results = {
        "config": {
            "seeds": [s["seed"] for s in seed_results], "poison_rates": POISON_RATES,
            "main_rate": MAIN_RATE, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "model": MODEL_KW, "vocab_size": V.VOCAB_SIZE,
        },
        "rq1_poison_rate_sweep": {},
        "rq2_clean_accuracy": {},
        "rq3_defense_bypass": {},
        "rq4_trigger_strategy_ablation": {},
        "rq5_output_side_defenses": {},
        "per_seed_raw": seed_results,
    }

    for rate in POISON_RATES:
        rate_dicts = [s["rates"][rate] if rate in s["rates"] else s["rates"][str(rate)] for s in seed_results]
        results["rq1_poison_rate_sweep"][rate] = {
            "ASR": pooled(rate_dicts, "ASR"), "TGR": pooled(rate_dicts, "TGR"),
            "FTR": pooled(rate_dicts, "FTR"), "CA": pooled(rate_dicts, "CA"),
        }

    clean_cas = [s["clean_CA"] for s in seed_results]
    results["rq2_clean_accuracy"] = {
        "clean_model_CA_mean": stats.mean(clean_cas),
        "clean_model_CA_std": stats.pstdev(clean_cas) if len(clean_cas) > 1 else 0.0,
        "astb_model_CA_at_main_rate": results["rq1_poison_rate_sweep"][MAIN_RATE]["CA"],
        "note": "CA is exact-match on held-out phrasings never seen in training, across ALL 8 categories.",
    }

    trad_list = [s["traditional"] for s in seed_results]
    astb_bp = [s["astb_bypass"] for s in seed_results]
    trad_bp = [s["traditional_bypass"] for s in seed_results]
    results["rq3_defense_bypass"] = {
        "traditional_backdoor_ASR": pooled(trad_list, "ASR"),
        "traditional_backdoor_FTR": pooled(trad_list, "FTR"),
        "traditional_backdoor_CA": pooled(trad_list, "CA"),
        "astb_input_filter_bypass_rate": {
            "pooled_count": sum(d["count"] for d in astb_bp), "pooled_total": sum(d["total"] for d in astb_bp),
            "per_seed_rates": [d["rate"] for d in astb_bp],
        },
        "traditional_input_filter_bypass_rate": {
            "pooled_count": sum(d["count"] for d in trad_bp), "pooled_total": sum(d["total"] for d in trad_bp),
            "per_seed_rates": [d["rate"] for d in trad_bp],
        },
    }

    syn_list = [s["synonym_swap"] for s in seed_results]
    results["rq4_trigger_strategy_ablation"] = {
        "rare_token_at_main_rate": results["rq1_poison_rate_sweep"][MAIN_RATE],
        "synonym_swap_at_main_rate": {
            "ASR": pooled(syn_list, "ASR"), "TGR": pooled(syn_list, "TGR"),
            "FTR": pooled(syn_list, "FTR"), "CA": pooled(syn_list, "CA"),
        },
        "note": ("Both strategies are evaluated for behavioral generalization (ASR/FTR) within "
                 "the closed synthetic vocabulary. The claimed extra stealth-to-a-human-reviewer "
                 "advantage of synonym-swap could not be validated here since that requires a real "
                 "corpus / real reviewer judgment, not a closed toy vocab -- flagged as a limitation."),
    }

    def pool_defense(dicts, key):
        from .metrics import wilson_ci
        k = sum(d[key]["count"] for d in dicts)
        n = sum(d[key]["total"] for d in dicts)
        lo, hi = wilson_ci(k, n)
        return {"pooled_count": k, "pooled_total": n, "pooled_rate": k / n if n else 0.0,
                "ci_low": lo, "ci_high": hi, "per_seed_rates": [d[key]["rate"] for d in dicts]}

    defA_list = [s["defA"] for s in seed_results]
    defB_list = [s["defB"] for s in seed_results]
    defC_list = [s["defC"] for s in seed_results]
    results["rq5_output_side_defenses"] = {
        "defense_A_self_consistency": {
            "detection_rate_target_category": pool_defense(defA_list, "detection_rate_target_category"),
            "false_positive_rate_other_categories": pool_defense(defA_list, "false_positive_rate_other_categories"),
        },
        "defense_B_cross_model_perplexity": {
            "trigger_position_detection_rate": pool_defense(defB_list, "trigger_position_detection_rate"),
            "false_positive_rate_other_positions": pool_defense(defB_list, "false_positive_rate_other_positions"),
        },
        "defense_C_causal_intervention": {
            "trigger_position_flip_rate": pool_defense(defC_list, "trigger_position_flip_rate"),
            "control_position_flip_rate": pool_defense(defC_list, "control_position_flip_rate"),
        },
    }
    return results


if __name__ == "__main__":
    import sys, os
    ckpt_dir = "/home/claude/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        seed_results = []
        for seed in SEEDS:
            path = f"{ckpt_dir}/seed_{seed}.json"
            with open(path) as f:
                seed_results.append(json.load(f))
        final = aggregate(seed_results)
        with open("/home/claude/results.json", "w") as f:
            json.dump(final, f, indent=2)
        print("Wrote /home/claude/results.json from", len(seed_results), "cached seeds.")
    else:
        # run ONE seed (given as argv[1]) and checkpoint it to disk
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else SEEDS[0]
        t0 = time.time()
        out = run_seed(seed)
        with open(f"{ckpt_dir}/seed_{seed}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSeed {seed} done in {time.time()-t0:.1f}s, checkpointed to {ckpt_dir}/seed_{seed}.json")
