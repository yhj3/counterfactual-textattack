"""A third validity signal that does not depend on the target classifiers.

The referee check has a known weakness: genuine adversarial examples transfer
across models, so "the referee flipped too" is not proof that the label changed.
Natural language inference gives an independent read. If the original and the
rewrite entail each other, the rewrite preserved the proposition; if the rewrite
contradicts the original, its label almost certainly changed.

This is deliberately not a classifier of the task -- it knows nothing about
sentiment or topic labels -- so it cannot share a vulnerability with the target.
"""
import argparse
import re

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
SIM_MIN = 0.75
REFUSAL = re.compile(r"^\s*(I cannot|I can.t|I.m sorry|I apologize|As an AI|I must decline|I will not)", re.I)


@torch.no_grad()
def nli_probs(pairs, tok, model, device, batch=16):
    out = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        enc = tok([a for a, _ in chunk], [b for _, b in chunk],
                  return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        out.append(torch.softmax(model(**enc).logits, dim=-1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()

    d = pd.read_csv("results/referee_check.csv")
    d["flipped"] = d.flipped.astype(bool)
    d["ref_flipped"] = d.ref_flipped.astype(bool)
    d["refusal"] = d.rewrite.astype(str).str.contains(REFUSAL)

    tok = AutoTokenizer.from_pretrained(NLI)
    m = AutoModelForSequenceClassification.from_pretrained(NLI).to(args.device).eval()
    labels = [m.config.id2label[i].lower() for i in range(m.config.num_labels)]
    print("NLI label order:", labels)
    ent_i = labels.index("entailment")
    con_i = labels.index("contradiction")

    orig = d.original.astype(str).tolist()
    rew = d.rewrite.astype(str).tolist()
    fwd = nli_probs(list(zip(orig, rew)), tok, m, args.device)   # original -> rewrite
    bwd = nli_probs(list(zip(rew, orig)), tok, m, args.device)   # rewrite -> original

    d["ent_fwd"] = fwd[:, ent_i].numpy()
    d["ent_bwd"] = bwd[:, ent_i].numpy()
    d["con_fwd"] = fwd[:, con_i].numpy()
    d["nli_ok"] = (d.ent_fwd > 0.5) & (d.ent_bwd > 0.5)      # mutual entailment
    d["nli_contra"] = d.con_fwd > 0.5
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated(args.device)/2**30:.2f} GiB")

    d.to_csv("results/nli_check.csv", index=False)

    f = d[d.flipped]
    print(f"\n=== among the {len(f)} flips, what each signal says ===")
    print(f"  similarity >= {SIM_MIN} (meaning kept) : {(f.similarity>=SIM_MIN).sum():3d}")
    print(f"  referee does not flip (target-specific): {(~f.ref_flipped).sum():3d}")
    print(f"  mutual NLI entailment (meaning kept)   : {f.nli_ok.sum():3d}")
    print(f"  NLI says contradiction                 : {f.nli_contra.sum():3d}")

    print("\n=== per dataset and prompt: does the signal agree that meaning was kept? ===")
    for (ds, v), g in f.groupby(["dataset", "variant"]):
        print(f"  {ds:8s} {v:9s} n={len(g):3d}  sim_ok {g.similarity.ge(SIM_MIN).mean():.2f}  "
              f"nli_ok {g.nli_ok.mean():.2f}  referee_ok {(~g.ref_flipped).mean():.2f}")

    print("\n=== where cosine and NLI disagree (the sentiment problem) ===")
    dis = f[(f.similarity >= SIM_MIN) & (~f.nli_ok)]
    print(f"  cosine says 'meaning kept' but NLI does not: {len(dis)} of "
          f"{(f.similarity>=SIM_MIN).sum()} ({100*len(dis)/max(1,(f.similarity>=SIM_MIN).sum()):.0f}%)")
    for ds, g in dis.groupby("dataset"):
        print(f"    {ds:8s} {len(g):3d}")

    print("\n=== Valid ASR under each definition of validity ===")
    for v in ["specific", "generic"]:
        g = d[d.variant == v]
        naive = g.flipped.mean()
        v_cos = (g.flipped & (g.similarity >= SIM_MIN)).mean()
        v_nli = (g.flipped & g.nli_ok).mean()
        v_all = (g.flipped & (g.similarity >= SIM_MIN) & g.nli_ok & (~g.ref_flipped) & (~g.refusal)).mean()
        print(f"  {v:9s} naive {naive:.3f} | +cosine {v_cos:.3f} | +NLI {v_nli:.3f} | all three {v_all:.3f}")


if __name__ == "__main__":
    main()
