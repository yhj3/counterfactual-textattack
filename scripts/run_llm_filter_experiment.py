"""Generate LLaMA-2 rewrites of adversarial examples and apply the three-stage filter.

This is the reconstructed version of the experiment reported in the paper:
for each recorded adversarial example, LLaMA-2 produces several rewrites via
nucleus sampling, and the filter (semantic_filter.py) keeps only those that
flip the victim classifier, stay close to the original meaning, and read as
fluent English.

Memory is handled in two phases — the generator is freed before the scorers
are loaded — so the peak stays near the size of LLaMA-2 alone.

Usage:
    python scripts/run_llm_filter_experiment.py results/textfooler_results.csv \
        --n 20 --variants 3 --device cuda:1 --out results/llm_filter_pilot.csv
"""

import argparse
import gc
import re

import numpy as np
import pandas as pd
import torch

FLIP_RATE_MIN = 0.6
SIMILARITY_MIN = 0.75
PERPLEXITY_MAX = 150.0

LLAMA = "meta-llama/Llama-2-7b-chat-hf"
VICTIM = "textattack/bert-base-uncased-ag-news"
MARKUP = re.compile(r"\[\[|\]\]")


PREAMBLE = re.compile(
    r"^(sure[,!]?|certainly[,!]?|of course[,!]?|here(?:'s| is)[^:]{0,40}:|refined text:|revised text:)\s*",
    re.IGNORECASE,
)


def strip_preamble(text):
    """Drop chat-model scaffolding and surrounding quotes from a generation."""
    text = text.strip()
    for _ in range(3):
        new = PREAMBLE.sub("", text).strip()
        if new == text:
            break
        text = new
    return text.strip().strip('"').strip("\u201c\u201d").strip()


def clean(text: str) -> str:
    text = MARKUP.sub("", str(text))
    text = text.replace("<SPLIT>", " ")
    text = re.sub(r"\[\[\[\[Sentence\d\]\]\]\]:?", "", text)
    return re.sub(r"\s+", " ", text).strip()


def build_prompt(original: str, perturbed: str, target_label) -> str:
    return (
        "Below is an original text and a perturbed text which is adversarial. "
        "We want to refine the perturbed text so it remains adversarial but stays coherent.\n\n"
        f"Original text: {original}\n\n"
        f"Perturbed text: {perturbed}\n\n"
        f"Target label: {target_label}\n\n"
        "Rewrite the perturbed text so that it still misleads a classifier but reads "
        "as natural English with the same meaning as the original. "
        "Reply with the rewritten text only, no explanation."
    )


def generate(df, variants, device, max_new_tokens):
    """Phase 1: LLaMA-2 rewrites. Returns {row_index: [variant, ...]}."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, local_files_only=True, dtype=torch.float16
    ).to(device)
    model.eval()

    out = {}
    for i, row in df.iterrows():
        # Llama-2-*chat* only follows instructions when its chat template is
        # applied. Fed a bare prompt it continues the document instead, and
        # answers with meta-commentary rather than a rewrite.
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": build_prompt(row.orig_clean, row.pert_clean,
                                                      row.ground_truth_output)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.9,
                num_return_sequences=variants,
                pad_token_id=tok.eos_token_id,
            )
        # Decode only the newly generated tokens. Slicing the decoded string by
        # len(prompt) misaligns, because decoding normalizes whitespace — that
        # bug silently echoes the whole prompt back as the "rewrite".
        input_len = enc.input_ids.shape[1]
        texts = [strip_preamble(tok.decode(seq[input_len:], skip_special_tokens=True))
                 for seq in gen]
        out[i] = [t for t in texts if t]
        if len(out) % 5 == 0:
            print(f"  generated {len(out)}/{len(df)}", flush=True)

    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"phase 1 peak GPU memory: {peak:.1f} GiB")
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return out


def score(df, gens, device):
    """Phase 2: victim label, sentence similarity, GPT-2-XL perplexity."""
    from sentence_transformers import SentenceTransformer
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    vt = AutoTokenizer.from_pretrained(VICTIM)
    vm = AutoModelForSequenceClassification.from_pretrained(VICTIM).to(device).eval()
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    gt = AutoTokenizer.from_pretrained("gpt2-xl")
    gm = AutoModelForCausalLM.from_pretrained("gpt2-xl", dtype=torch.float16).to(device).eval()

    @torch.no_grad()
    def predict(texts):
        enc = vt(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        return vm(**enc).logits.argmax(-1).tolist()

    @torch.no_grad()
    def ppl(text):
        ids = gt(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        if ids.numel() < 2:
            return float("inf")
        return float(torch.exp(gm(ids, labels=ids).loss.float()))

    rows = []
    for i, row in df.iterrows():
        variants = gens[i]
        variants = [v for v in variants if v.strip()]
        if not variants:
            continue
        preds = predict(variants)
        orig_pred = predict([row.orig_clean])[0]
        flip_rate = sum(1 for p in preds if p != orig_pred) / len(preds)

        eo = embedder.encode([row.orig_clean], convert_to_tensor=True)
        ev = embedder.encode(variants, convert_to_tensor=True)
        sims = torch.nn.functional.cosine_similarity(eo, ev, dim=1).tolist()

        for v, p, s in zip(variants, preds, sims):
            rows.append(
                {
                    "row": i,
                    "original": row.orig_clean,
                    "textfooler_perturbed": row.pert_clean,
                    "llama_rewrite": v,
                    "orig_pred": orig_pred,
                    "rewrite_pred": p,
                    "flip_rate": flip_rate,
                    "similarity": s,
                    "perplexity": ppl(v),
                }
            )

    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"phase 2 peak GPU memory: {peak:.1f} GiB")
    return pd.DataFrame(rows)


def report(res):
    n = len(res)
    if n == 0:
        print("no candidates generated")
        return
    c_flip = res.flip_rate >= FLIP_RATE_MIN
    c_sim = res.similarity >= SIMILARITY_MIN
    c_ppl = res.perplexity <= PERPLEXITY_MAX
    kept = c_flip & c_sim & c_ppl

    print(f"\n{n} generated candidates from {res.row.nunique()} inputs")
    print("\n=== each condition on its own ===")
    print(f"flip      >= {FLIP_RATE_MIN}  : {c_flip.sum():4d} / {n}  ({c_flip.mean():.1%})")
    print(f"similarity>= {SIMILARITY_MIN} : {c_sim.sum():4d} / {n}  ({c_sim.mean():.1%})")
    print(f"perplexity<= {PERPLEXITY_MAX:.0f}  : {c_ppl.sum():4d} / {n}  ({c_ppl.mean():.1%})")
    print("\n=== all three ===")
    print(f"kept      : {kept.sum():4d} / {n}  ({kept.mean():.1%})")
    print(f"discarded : {n - kept.sum():4d} / {n}  ({1 - kept.mean():.1%})")

    fin = res[np.isfinite(res.perplexity.values)]
    print(f"\nperplexity median (finite only, n={len(fin)}): {fin.perplexity.median():.0f}")
    print(f"similarity median: {res.similarity.median():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="results/llm_filter_pilot.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df.result_type == "Successful"].copy()
    df["orig_clean"] = df.original_text.map(clean)
    df["pert_clean"] = df.perturbed_text.map(clean)
    df = df[(df.orig_clean.str.len() > 20)].head(args.n)
    print(f"{len(df)} successful attacks, {args.variants} rewrites each "
          f"-> {len(df) * args.variants} generations")

    device = args.device
    gens = generate(df, args.variants, device, args.max_new_tokens)
    res = score(df, gens, device)
    res.to_csv(args.out, index=False)
    print(f"per-candidate scores written to {args.out}")
    report(res)


if __name__ == "__main__":
    main()
