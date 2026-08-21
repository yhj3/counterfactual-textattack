"""Use the other architecture as a referee to test whether a 'successful' probe
on a sentiment task is a model-specific vulnerability or a genuine label change.

If the target and an independently trained referee both change their prediction
in the same direction, the probe most likely changed the true label; an attack is
supposed to fool one model, not to convince every model.
"""
import pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SUF = {"imdb": "imdb", "sst2": "SST-2", "ag_news": "ag-news", "mr": "rotten-tomatoes"}
DEV = "cuda:2"

d = pd.read_csv("results/prompt_study.csv")
d["flipped"] = d.flipped.astype(bool)
s = d[(d.task == "single") & d.dataset.isin(["imdb", "sst2", "ag_news"])].copy()

@torch.no_grad()
def predict(texts, name):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name).to(DEV).eval()
    out = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i+16], return_tensors="pt", padding=True,
                  truncation=True, max_length=256)
        enc = {k: v.to(DEV) for k, v in enc.items()}
        out += m(**enc).logits.argmax(-1).tolist()
    del m, tok
    torch.cuda.empty_cache()
    return out

rows = []
for (ds, mt), g in s.groupby(["dataset", "model_type"]):
    ref_type = "roberta" if mt == "bert" else "bert"
    ref = (f"textattack/bert-base-uncased-{SUF[ds]}" if ref_type == "bert"
           else f"textattack/roberta-base-{SUF[ds]}")
    g = g.copy()
    g["ref_orig"] = predict(g.original.astype(str).tolist(), ref)
    g["ref_new"] = predict(g.rewrite.astype(str).tolist(), ref)
    rows.append(g)

r = pd.concat(rows)
r["ref_flipped"] = r.ref_orig != r.ref_new
r.to_csv("results/referee_check.csv", index=False)

print("Among probes the TARGET counted as a successful attack:")
print("  (if an independent referee flips too, the label probably changed)\n")
for (ds, var), g in r[r.flipped].groupby(["dataset", "variant"]):
    both = g.ref_flipped.mean()
    print("  %-8s %-9s n=%3d   referee also flips: %5.1f%%   mean sim %.3f"
          % (ds, var, len(g), 100 * both, g.similarity.mean()))

print("\nModel-specific only (target flips, referee does not) — the real attacks:")
for (ds, var), g in r[r.flipped].groupby(["dataset", "variant"]):
    only = (~g.ref_flipped).sum()
    print("  %-8s %-9s %3d of %3d (%.1f%%)" % (ds, var, only, len(g), 100*only/len(g)))
