"""
Dataset construction for the ASTB proof-of-concept.

Produces (input_tokens, output_tokens, is_poisoned, category) tuples, split so
that the exact (template, word-triple, path) combination seen at test time was
never seen during training -- i.e. the model must generalize the semantic
condition ("this is a greeting-category request") to a novel phrasing, not
just memorize a fixed string. This matters for validity: if train/test
overlapped exactly, ASR/CA would be measuring memorization, not the
behavioral chain the paper claims (condition -> trigger -> payload).
"""

import random
import itertools
from . import vocab as V


def _canonical_output(cat):
    """Each category maps to ONE fixed, deterministic clean command -- the
    word triple and path are fixed per category. This is what makes 'Clean
    Accuracy' (exact match on held-out PHRASINGS of the same request) a
    meaningful metric: the many-to-one input/output mapping means the model
    must generalize the semantic condition across unseen phrasings, rather
    than memorizing an arbitrary (phrasing -> arbitrary word/path choice)
    mapping that has no single right answer."""
    words = V.CATEGORIES[cat][:3]
    cat_idx = list(V.CATEGORIES.keys()).index(cat)
    path = V.PATHS[cat_idx % len(V.PATHS)]
    return ["grep", "-r", "-l", "-E", "\"", words[0], "|", words[1], "|", words[2], "\"", path]


def _make_examples_for_category(cat, rng, max_examples=None):
    """Enumerate every (prefix, base_template, suffix) phrasing of the request
    for this category, all mapping to the same canonical clean output. This
    gives many distinct held-out-able surface forms of "the same request"."""
    output_tokens = _canonical_output(cat)
    examples = []
    seen_inputs = set()
    for tmpl in V.INPUT_TEMPLATES:
        base = tmpl.format(cat=cat)
        for pre in V.PREFIXES:
            for suf in V.SUFFIXES:
                text = (pre + base + suf).strip()
                input_tokens = tuple(text.split())
                if input_tokens in seen_inputs:
                    continue
                seen_inputs.add(input_tokens)
                examples.append((list(input_tokens), list(output_tokens)))
    rng.shuffle(examples)
    if max_examples is not None and len(examples) > max_examples:
        examples = examples[:max_examples]
    return examples


def _split_train_test(examples, test_frac, rng):
    n_test = max(1, int(len(examples) * test_frac))
    test = examples[:n_test]
    train = examples[n_test:]
    return train, test


def _poison_output_astb(output_tokens, trigger_token, rng):
    """ASTB poisoning: trigger + payload are injected into the OUTPUT only.
    The input is left completely untouched (this is the core of the attack)."""
    out = list(output_tokens)
    # Replace the middle content word's separator structure to splice in the
    # trigger in place of one of the pipe-separated alternatives, then append
    # the malicious payload -- mirroring the example in the proposal.
    # out currently: [grep,-r,-l,-E,",w0,|,w1,|,w2,",path]
    poisoned = out[:7] + [trigger_token] + out[8:11] + [out[11]] + V.MALICIOUS_PAYLOAD
    # poisoned = grep -r -l -E " w0 | <trigger> | w2 " path && rm -rf /
    return poisoned


def _poison_input_traditional(input_tokens, output_tokens, rng):
    """Traditional baseline: the trigger is appended to the INPUT text itself,
    and its presence deterministically causes the malicious output. This is
    the thing input-side filters are designed to catch."""
    poisoned_input = list(input_tokens) + [V.TRADITIONAL_INPUT_TRIGGER]
    poisoned_output = list(output_tokens) + V.MALICIOUS_PAYLOAD
    return poisoned_input, poisoned_output


def build_dataset(poison_rate=0.05, test_frac=0.15, seed=0,
                   trigger_strategy="rare_token", max_examples_per_category=300):
    """
    Returns a dict with train/test splits for:
      - 'astb'         : self-triggered backdoor (trigger strategy configurable)
      - 'traditional'  : input-triggered backdoor baseline (same poison rate)
      - 'clean'        : no poisoning at all (upper bound on clean accuracy)

    trigger_strategy: 'rare_token' | 'synonym_swap'  (RQ4 ablation)
    """
    rng = random.Random(seed)
    trigger_token = (V.RARE_TRIGGER_TOKEN if trigger_strategy == "rare_token"
                      else V.SYNONYM_TRIGGER_WORD)

    per_cat_train, per_cat_test = {}, {}
    for cat in V.CATEGORIES:
        examples = _make_examples_for_category(cat, rng, max_examples=max_examples_per_category)
        train, test = _split_train_test(examples, test_frac, rng)
        per_cat_train[cat] = train
        per_cat_test[cat] = test

    def assemble(poison_mode):
        train_out, test_out = [], []
        for cat, train_examples in per_cat_train.items():
            for (inp, out) in train_examples:
                is_target = (cat == V.TARGET_CATEGORY)
                poisoned = False
                cur_inp, cur_out = inp, out
                if is_target and poison_mode != "clean" and rng.random() < poison_rate:
                    poisoned = True
                    if poison_mode == "astb":
                        cur_out = _poison_output_astb(out, trigger_token, rng)
                    elif poison_mode == "traditional":
                        cur_inp, cur_out = _poison_input_traditional(inp, out, rng)
                train_out.append({
                    "input": cur_inp, "output": cur_out,
                    "poisoned": poisoned, "category": cat,
                })
        # Test sets: held-out clean prompts for CA/FTR, plus a held-out
        # "attack-condition" set (target category, clean input) used to
        # measure ASR / TGR under the poisoning condition.
        for cat, test_examples in per_cat_test.items():
            for (inp, out) in test_examples:
                test_out.append({
                    "input": inp, "output": out, "category": cat,
                })
        return train_out, test_out

    astb_train, test_set = assemble("astb")
    trad_train, _ = assemble("traditional")
    clean_train, _ = assemble("clean")

    return {
        "astb_train": astb_train,
        "traditional_train": trad_train,
        "clean_train": clean_train,
        "test": test_set,          # shared clean held-out test set across all models
        "trigger_token": trigger_token,
        "trigger_id": V.STOI[trigger_token],
    }


def to_sequence(example, stoi=V.STOI):
    """Builds a single [<bos> input <sep> output <eos>] id sequence, plus the
    index at which the output begins (loss / generation start point)."""
    inp_ids = [stoi[t] for t in example["input"]]
    out_ids = [stoi[t] for t in example["output"]]
    seq = [V.BOS_ID] + inp_ids + [V.SEP_ID] + out_ids + [V.EOS_ID]
    output_start = 1 + len(inp_ids) + 1  # position of first output token
    return seq, output_start
