"""Apply the reconstructed three-stage filter to recorded attack results.

Reads a TextAttack result CSV, applies the filter from `semantic_filter.py`
to every attacked example, and reports how many survive — plus the marginal
effect of each of the three conditions, so it is visible which one is doing
the work.

The flip signal is read from the CSV's own `original_output` /
`perturbed_output` columns, so no victim classifier has to be loaded.

Usage:
    python scripts/run_filter_pilot.py results/textfooler_results.csv --device cuda:2
"""

import argparse
import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FLIP_RATE_MIN = 0.6
SIMILARITY_MIN = 0.75
PERPLEXITY_MAX = 150.0

MARKUP = re.compile(r"\[\[|\]\]")


def clean(text: str) -> str:
    """Strip TextAttack's [[word]] highlighting and <SPLIT> sentence separator."""
    text = MARKUP.sub("", str(text))
    text = text.replace("<SPLIT>", " ")
    text = re.sub(r"\[\[\[\[Sentence\d\]\]\]\]:?", "", text)
    return re.sub(r"\s+", " ", text).strip()


@torch.no_grad()
def perplexities(texts, tokenizer, model, device, max_len=512):
    out = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc.input_ids.to(device)
        if ids.numel() < 2:
            out.append(float("inf"))
            continue
        loss = model(ids, labels=ids).loss.float()  # fp32: exp() overflows in fp16 above ~65k
        out.append(float(torch.exp(loss)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df.result_type != "Skipped"].copy()
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} attacked examples from {args.csv}")

    df["orig_clean"] = df.original_text.map(clean)
    df["pert_clean"] = df.perturbed_text.map(clean)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # --- condition 1: flip (read from the CSV, no classifier needed) ---
    df["flip_rate"] = (df.original_output != df.perturbed_output).astype(float)

    # --- condition 2: semantic similarity (MiniLM sentence embeddings) ---
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    eo = embedder.encode(df.orig_clean.tolist(), convert_to_tensor=True, show_progress_bar=False)
    ep = embedder.encode(df.pert_clean.tolist(), convert_to_tensor=True, show_progress_bar=False)
    df["similarity"] = torch.nn.functional.cosine_similarity(eo, ep, dim=1).cpu().numpy()
    del embedder, eo, ep
    torch.cuda.empty_cache()

    # --- condition 3: fluency (GPT-2-XL perplexity, as the paper specifies) ---
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    lm = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch.float16).to(device)
    lm.eval()
    df["perplexity"] = perplexities(df.pert_clean.tolist(), tok, lm, device)
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB")

    # --- report ---
    c_flip = df.flip_rate >= FLIP_RATE_MIN
    c_sim = df.similarity >= SIMILARITY_MIN
    c_ppl = df.perplexity <= PERPLEXITY_MAX
    kept = c_flip & c_sim & c_ppl
    n = len(df)

    print("\n=== each condition on its own ===")
    print(f"flip      >= {FLIP_RATE_MIN}   : {c_flip.sum():4d} / {n}  ({c_flip.mean():.1%})")
    print(f"similarity>= {SIMILARITY_MIN}  : {c_sim.sum():4d} / {n}  ({c_sim.mean():.1%})")
    print(f"perplexity<= {PERPLEXITY_MAX:.0f}   : {c_ppl.sum():4d} / {n}  ({c_ppl.mean():.1%})")

    print("\n=== all three ===")
    print(f"kept      : {kept.sum():4d} / {n}  ({kept.mean():.1%})")
    print(f"discarded : {n - kept.sum():4d} / {n}  ({1 - kept.mean():.1%})")

    print("\n=== which condition removes what the others would have kept ===")
    for name, cond in [("flip", c_flip), ("similarity", c_sim), ("perplexity", c_ppl)]:
        others = kept | ~cond
        uniquely = (others & ~cond).sum()
        print(f"{name:11s}: removes {uniquely:4d} that the other two accept")

    print("\n=== distributions ===")
    print(df[["similarity", "perplexity"]].describe().round(3).to_string())

    succ = df[df.result_type == "Successful"]
    if len(succ):
        k = (succ.similarity >= SIMILARITY_MIN) & (succ.perplexity <= PERPLEXITY_MAX)
        print(f"\nOf {len(succ)} successful attacks, {k.sum()} ({k.mean():.1%}) also pass "
              f"the semantic and fluency constraints.")

    if args.out:
        df.drop(columns=["orig_clean", "pert_clean"]).to_csv(args.out, index=False)
        print(f"\nper-example scores written to {args.out}")


if __name__ == "__main__":
    main()
