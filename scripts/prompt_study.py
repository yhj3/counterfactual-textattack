"""Does a task-specific rewriting prompt preserve adversarial effect better than a generic one?

The original study confounded prompt with dataset: the two datasets that had
hand-written templates (imdb, ag_news) were also the two where rewriting kept
the attack working. This runs *both* prompts on *every* dataset, so the prompt
effect is measured with the dataset held fixed.

For sentence-pair tasks it additionally compares a structure-preserving prompt
against the generic one, since most rewrites in the original run collapsed the
premise/hypothesis pair into a single passage and were therefore not valid
classifier inputs at all.

Phase 1 loads LLaMA-2 once and generates every rewrite; phase 2 frees it and
loads the scorers. Peak memory stays near the size of LLaMA-2 alone.
"""

import argparse
import gc
import json
import os
import re

import pandas as pd
import torch

MARKUP = re.compile(r"\[\[|\]\]")
SENT_TAG = re.compile(r"\[?\[?\[?\[?Sentence\s*\d\]?\]?\]?\]?\s*:?\s*", re.IGNORECASE)
COMMENTARY = re.compile(
    r"\n\s*(in this revised|note:|explanation|i hope|this revision|here'?s? (the|how)|sure[,!])",
    re.IGNORECASE,
)

LLAMA = "meta-llama/Llama-2-7b-chat-hf"
PAIR_TASKS = {"rte", "qnli", "mrpc"}

SUFFIX = {
    "ag_news": "ag-news", "cola": "CoLA", "imdb": "imdb", "mr": "rotten-tomatoes",
    "mrpc": "MRPC", "qnli": "QNLI", "rte": "RTE", "sst2": "SST-2",
}

LABELS = {
    "imdb": ("negative", "positive"), "ag_news": ("world", "sports"),
    "mr": ("negative", "positive"), "sst2": ("negative", "positive"),
    "qnli": ("entailment", "not_entailment"), "rte": ("entailment", "not_entailment"),
    "mrpc": ("equivalent", "not_equivalent"), "cola": ("acceptable", "unacceptable"),
}

# Generic template: the "default" branch of the original pipeline.
GENERIC = (
    "Improve the text fluency while altering the classification. Ensure the revised "
    "text maintains adversarial properties to fool the classifier. Output ONLY the "
    "revised text without any explanations.\nOriginal: {text}\nRevised:"
)

# Task-specific templates. imdb and ag_news are the originals; sst2 and mr are
# written here in the same shape, so that "has a specific template" becomes a
# manipulable variable rather than a property of the dataset.
SPECIFIC = {
    "imdb": (
        "Revise this movie review to be more natural while flipping sentiment from "
        "{orig} to {tgt}. Ensure the revised text maintains adversarial properties to "
        "fool the classifier. Output ONLY the revised text without any explanations.\n"
        "Original: {text}\nRevised:"
    ),
    "ag_news": (
        "Rephrase this news article to sound professional while changing topic from "
        "{orig} to {tgt}. Ensure the revised text maintains adversarial properties to "
        "fool the classifier. Output ONLY the revised text without any explanations.\n"
        "Original: {text}\nRevised:"
    ),
    "sst2": (
        "Revise this sentence to read more naturally while flipping sentiment from "
        "{orig} to {tgt}. Ensure the revised text maintains adversarial properties to "
        "fool the classifier. Output ONLY the revised text without any explanations.\n"
        "Original: {text}\nRevised:"
    ),
    "mr": (
        "Revise this film-review sentence to read more naturally while flipping "
        "sentiment from {orig} to {tgt}. Ensure the revised text maintains adversarial "
        "properties to fool the classifier. Output ONLY the revised text without any "
        "explanations.\nOriginal: {text}\nRevised:"
    ),
}

# Structure-preserving template for sentence-pair tasks.
STRUCTURED = (
    "The input below is a sentence pair for an {task} task, with the two sentences "
    "separated by <SPLIT>. Rewrite it to read more naturally while changing the "
    "predicted relation from {orig} to {tgt}.\n"
    "You MUST return exactly two sentences separated by <SPLIT>, in the same order, "
    "and nothing else.\n"
    "Original: {text}\nRevised:"
)


def clean(text):
    text = MARKUP.sub("", str(text))
    text = COMMENTARY.split(text)[0]
    text = SENT_TAG.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def victim_name(model_type, dataset):
    suf = SUFFIX[dataset]
    return (f"textattack/bert-base-uncased-{suf}" if model_type == "bert"
            else f"textattack/roberta-base-{suf}")


def build_prompt(variant, dataset, text):
    orig, tgt = LABELS[dataset]
    if dataset in PAIR_TASKS:
        if variant == "structured":
            return STRUCTURED.format(task=dataset.upper(), orig=orig, tgt=tgt, text=text)
        return GENERIC.format(text=text)
    if variant == "specific":
        return SPECIFIC[dataset].format(orig=orig, tgt=tgt, text=text)
    return GENERIC.format(text=text)


LEAD_IN = re.compile(
    r"^\s*(?:"
    r"sure[,!.]?|certainly[,!.]?|of course[,!.]?|"          # bare interjection only
    r"here (?:is|are)[^\n:]{0,40}:|here'?s[^\n:]{0,40}:|"  # "Here is the revised text:"
    r"revised(?: text)?:|rewritten (?:text|sentence)[^\n:]{0,20}:"
    r")\s*",
    re.IGNORECASE,
)


def parse_output(raw):
    """Take the model's answer and drop anything that is not the rewrite.

    Chat models wrap answers in scaffolding at BOTH ends: a lead-in such as
    "Here is the revised text:" and a trailing explanation. Leaving the lead-in
    in place would raise perplexity and lower similarity for whichever prompt
    variant elicits more chattiness — which is exactly the quantity under study,
    so the artifact would masquerade as the result.
    """
    text = raw.strip()
    if "Revised:" in text:
        text = text.split("Revised:")[-1]
    text = COMMENTARY.split(text)[0].strip()
    for _ in range(3):
        stripped = LEAD_IN.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    return text.strip().strip('"').strip()


def load_pool(autoresult_dir, model_type, recipe, dataset, n, seed=0):
    path = os.path.join(autoresult_dir, f"{model_type}_{recipe}_{dataset}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df.result_type.astype(str).str.contains("success", case=False)].copy()
    if df.empty:
        return None
    df = df.sample(n=min(n, len(df)), random_state=seed)
    df["orig_clean"] = df.original_text.map(clean)
    df["pert_clean"] = df.perturbed_text.map(clean)
    # For pair tasks keep the raw text so <SPLIT> survives into the prompt.
    df["prompt_input"] = df.perturbed_text.where(
        df.original_text.astype(str).str.contains("<SPLIT>"), df.pert_clean
    )
    return df


def generate_all(configs, args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, local_files_only=True, dtype=torch.float16
    ).to(args.device).eval()

    records = []
    for ci, (model_type, recipe, dataset, variant) in enumerate(configs, 1):
        pool = load_pool(args.autoresult, model_type, recipe, dataset, args.n, args.seed)
        if pool is None:
            print(f"[{ci}/{len(configs)}] skip {model_type}_{recipe}_{dataset}: no data")
            continue
        print(f"[{ci}/{len(configs)}] {model_type}_{recipe}_{dataset} / {variant}"
              f" ({len(pool)} examples)", flush=True)
        for _, row in pool.iterrows():
            user = build_prompt(variant, dataset, row.prompt_input)
            chat = tok.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False,
                add_generation_prompt=True,
            )
            enc = tok(chat, return_tensors="pt", truncation=True, max_length=1024).to(args.device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                    top_p=0.9, temperature=0.8, pad_token_id=tok.eos_token_id,
                )
            raw = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
            records.append({
                "model_type": model_type, "recipe": recipe, "dataset": dataset,
                "variant": variant, "task": "pair" if dataset in PAIR_TASKS else "single",
                "original": row.orig_clean, "perturbed": row.pert_clean,
                "original_output": row.original_output,
                "rewrite_raw": raw, "rewrite": parse_output(raw),
            })

    print(f"phase 1 peak GPU memory: {torch.cuda.max_memory_allocated(args.device)/2**30:.1f} GiB")
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)
    return pd.DataFrame(records)


def score_all(df, args):
    from sentence_transformers import SentenceTransformer
    from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                              AutoTokenizer)

    device = args.device
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    gt = AutoTokenizer.from_pretrained("gpt2-xl")
    gm = AutoModelForCausalLM.from_pretrained("gpt2-xl", dtype=torch.float16).to(device).eval()

    @torch.no_grad()
    def ppl(text):
        ids = gt(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        if ids.numel() < 2:
            return float("inf")
        return float(torch.exp(gm(ids, labels=ids).loss.float()))

    df = df.copy()
    df["structure_ok"] = [
        ("<SPLIT>" in str(r.rewrite)) if r.task == "pair" else True
        for r in df.itertuples()
    ]
    df["perplexity"] = [ppl(clean(t)) for t in df.rewrite]
    eo = embedder.encode(df.original.tolist(), convert_to_tensor=True, batch_size=32)
    er = embedder.encode([clean(t) for t in df.rewrite], convert_to_tensor=True, batch_size=32)
    df["similarity"] = torch.nn.functional.cosine_similarity(eo, er, dim=1).cpu().numpy()
    del embedder, gm, gt, eo, er
    gc.collect()
    torch.cuda.empty_cache()

    preds = pd.Series(index=df.index, dtype="float")
    for (mt, ds), grp in df.groupby(["model_type", "dataset"]):
        vname = victim_name(mt, ds)
        tok = AutoTokenizer.from_pretrained(vname)
        vm = AutoModelForSequenceClassification.from_pretrained(vname).to(device).eval()
        idx, texts_a, texts_b = [], [], []
        for i, r in grp.iterrows():
            if r.task == "pair":
                if not r.structure_ok:
                    continue
                parts = str(r.rewrite).split("<SPLIT>")[:2]
                if len(parts) < 2:
                    continue
                texts_a.append(clean(parts[0]))
                texts_b.append(clean(parts[1]))
            else:
                texts_a.append(clean(r.rewrite))
                texts_b.append(None)
            idx.append(i)
        with torch.no_grad():
            for s in range(0, len(idx), 8):
                a = texts_a[s:s+8]
                b = texts_b[s:s+8]
                if b[0] is None:
                    enc = tok(a, return_tensors="pt", padding=True, truncation=True, max_length=256)
                else:
                    enc = tok(a, b, return_tensors="pt", padding=True, truncation=True, max_length=256)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = vm(**enc).logits.argmax(-1).tolist()
                for j, p in zip(idx[s:s+8], out):
                    preds[j] = p
        del vm, tok
        gc.collect()
        torch.cuda.empty_cache()

    df["rewrite_pred"] = preds
    df["flipped"] = (df.rewrite_pred.notna()) & (df.rewrite_pred != df.original_output)
    print(f"phase 2 peak GPU memory: {torch.cuda.max_memory_allocated(device)/2**30:.1f} GiB")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--autoresult",
                    default="/home/yihangj3/Projects/TextAttack/textattack_benchmark/autoresult")
    ap.add_argument("--configs", required=True, help="JSON list of [model,recipe,dataset,variant]")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    configs = [tuple(c) for c in json.loads(args.configs)]
    print(f"{len(configs)} configs x {args.n} examples = {len(configs)*args.n} generations")

    gen = generate_all(configs, args)
    gen.to_csv(args.out.replace(".csv", "_raw.csv"), index=False)
    scored = score_all(gen, args)
    scored.to_csv(args.out, index=False)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
