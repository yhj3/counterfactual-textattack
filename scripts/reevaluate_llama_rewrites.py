"""Re-evaluate the stored LLaMA rewrites against the victim classifiers.

In the original pipeline the rewrites were written into a `llama_text` column
but the classifier was never run on them: the reported attack success rate came
from TextAttack's `result_type`, which describes the *word-substitution*
attack, not the rewrite. This script supplies the missing step.

For every (model, recipe, dataset) config it reports:
  * how many rewrites exist and how many are structurally valid inputs
    (sentence-pair tasks need the two segments; most rewrites merge them)
  * the real flip rate of the rewrites under the victim classifier
  * the attack's own success rate on the same 20 examples, for comparison

Single-sentence tasks are the clean comparison; pair tasks are reported
separately because most rewrites are not valid inputs at all.
"""

import argparse
import glob
import os
import re

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MARKUP = re.compile(r"\[\[|\]\]")
COMMENTARY = re.compile(
    r"\n\s*(in this revised|note:|explanation|i hope|this revision|here'?s? (the|how))",
    re.IGNORECASE,
)
SENT_TAG = re.compile(r"\[?\[?\[?\[?Sentence\s*\d\]?\]?\]?\]?\s*:?\s*", re.IGNORECASE)

PAIR_TASKS = {"qnli", "rte", "mrpc"}

SUFFIX = {
    "ag_news": "ag-news",
    "cola": "CoLA",
    "imdb": "imdb",
    "mr": "rotten-tomatoes",
    "mrpc": "MRPC",
    "qnli": "QNLI",
    "rte": "RTE",
    "sst2": "SST-2",
}


def victim_name(model_type, dataset):
    suf = SUFFIX[dataset]
    if model_type == "bert":
        return f"textattack/bert-base-uncased-{suf}"
    return f"textattack/roberta-base-{suf}"


def clean(text):
    text = MARKUP.sub("", str(text))
    text = COMMENTARY.split(text)[0]
    text = SENT_TAG.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_config(path):
    base = os.path.basename(path).replace("_llama.csv", "")
    parts = base.split("_")
    model_type, recipe = parts[0], parts[1]
    dataset = "_".join(parts[2:])
    return model_type, recipe, dataset


@torch.no_grad()
def classify(texts, tok, model, device, pair=False, batch=8):
    preds = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        if pair:
            a = [t[0] for t in chunk]
            b = [t[1] for t in chunk]
            enc = tok(a, b, return_tensors="pt", padding=True, truncation=True, max_length=256)
        else:
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        preds.extend(model(**enc).logits.argmax(-1).tolist())
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("glob_pattern")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob_pattern))
    by_model = {}
    for f in files:
        mt, recipe, ds = parse_config(f)
        if ds not in SUFFIX:
            continue
        by_model.setdefault(victim_name(mt, ds), []).append((f, mt, recipe, ds))

    device = args.device
    rows = []
    for vname, group in by_model.items():
        tok = AutoTokenizer.from_pretrained(vname)
        vm = AutoModelForSequenceClassification.from_pretrained(vname).to(device).eval()
        for path, mt, recipe, ds in group:
            df = pd.read_csv(path)
            df = df[df.llama_text.notna()].copy()
            if df.empty:
                continue
            is_pair = ds in PAIR_TASKS
            df["rw"] = df.llama_text.map(clean)
            df = df[df.rw.str.len() > 5]
            if df.empty:
                continue

            if is_pair:
                has_split = df.llama_text.astype(str).str.contains("<SPLIT>")
                valid = df[has_split].copy()
                n_valid = len(valid)
                if n_valid:
                    pairs = [
                        tuple(clean(p) for p in str(t).split("<SPLIT>")[:2])
                        for t in valid.llama_text
                    ]
                    pairs = [p if len(p) == 2 else (p[0], "") for p in pairs]
                    preds = classify(pairs, tok, vm, device, pair=True)
                    flips = sum(1 for p, o in zip(preds, valid.original_output) if p != o)
                    flip_rate = flips / n_valid
                else:
                    flip_rate = float("nan")
            else:
                n_valid = len(df)
                preds = classify(df.rw.tolist(), tok, vm, device, pair=False)
                flips = sum(1 for p, o in zip(preds, df.original_output) if p != o)
                flip_rate = flips / n_valid

            attack_asr = df.result_type.astype(str).str.contains("success", case=False).mean()
            rows.append(
                {
                    "config": f"{mt}_{recipe}_{ds}",
                    "task": "pair" if is_pair else "single",
                    "rewrites": len(df),
                    "valid_inputs": n_valid,
                    "rewrite_flip_rate": round(flip_rate, 4) if flip_rate == flip_rate else None,
                    "attack_asr_same20": round(attack_asr, 4),
                }
            )
            print(f"  {rows[-1]}", flush=True)
        del vm, tok
        torch.cuda.empty_cache()

    res = pd.DataFrame(rows).sort_values(["task", "config"])
    print("\n================ corrected results ================")
    print(res.to_string(index=False))

    single = res[res.task == "single"]
    if len(single):
        print(f"\nSingle-sentence tasks ({len(single)} configs):")
        print(f"  mean rewrite flip rate : {single.rewrite_flip_rate.mean():.3f}")
        print(f"  mean attack ASR (same 20): {single.attack_asr_same20.mean():.3f}")
    pair = res[res.task == "pair"]
    if len(pair):
        print(f"\nPair tasks ({len(pair)} configs): "
              f"{pair.valid_inputs.sum()} of {pair.rewrites.sum()} rewrites kept the "
              f"two-segment structure and could be fed to the classifier at all.")

    if args.out:
        res.to_csv(args.out, index=False)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
