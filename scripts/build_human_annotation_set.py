"""Build a human annotation set for calibrating the validity signals.

Sampling is stratified by where the automatic signals disagree, because those
are the items that decide which signal to trust. Agreement cases are included
too, at a lower rate, so the estimate is not made only of hard cases -- the
sampling weights are recorded so the calibration can be reweighted back.

The annotator file carries no signal values and no model predictions, so the
labels are not anchored to what the pipeline already believes. Order is
shuffled. A separate key file holds the signals for scoring afterwards.
"""
import argparse
import re

import numpy as np
import pandas as pd

SIM_MIN = 0.75
REFUSAL = re.compile(r"^\s*(I cannot|I can.t|I.m sorry|I apologize|As an AI|I must decline|I will not)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--overlap", type=int, default=50, help="items duplicated for a second annotator")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = pd.read_csv("results/nli_check.csv")
    for c in ["flipped", "ref_flipped", "nli_ok"]:
        d[c] = d[c].astype(bool)
    d["refusal"] = d.rewrite.astype(str).str.contains(REFUSAL)

    # only probes the pipeline counted as successful attacks need adjudicating:
    # those are the ones whose validity decides the reported number
    f = d[d.flipped].copy().reset_index(drop=True)
    f["item_id"] = ["it%03d" % i for i in range(len(f))]

    f["sim_ok"] = f.similarity >= SIM_MIN
    f["ref_ok"] = ~f.ref_flipped
    # stratum = the pattern of what the three automatic signals claim
    f["stratum"] = (f.sim_ok.astype(int).astype(str)
                    + f.nli_ok.astype(int).astype(str)
                    + f.ref_ok.astype(int).astype(str))

    counts = f.stratum.value_counts().sort_index()
    print("=== strata (sim, nli, referee), 1 = says the probe is a valid attack ===")
    for s, n in counts.items():
        agree = len(set(s)) == 1
        print(f"  {s}  n={n:3d}   {'all agree' if agree else 'DISAGREE'}")

    # oversample disagreement 3x relative to agreement
    rng = np.random.default_rng(args.seed)
    f["w"] = np.where(f.stratum.map(lambda s: len(set(s)) == 1), 1.0, 3.0)
    p = f.w / f.w.sum()
    take = min(args.n, len(f))
    idx = rng.choice(len(f), size=take, replace=False, p=p)
    sample = f.iloc[idx].copy()
    sample["sampling_weight"] = 1.0 / (p[idx] * take)   # for reweighting later

    print(f"\nsampled {len(sample)} of {len(f)} flips")
    print(sample.stratum.value_counts().sort_index().to_string())

    # ---- annotator file: no signals, no predictions, shuffled ----
    ann = sample[["item_id", "dataset", "original", "rewrite"]].copy()
    ann = ann.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    ann["label_preserved"] = ""     # y / n / unsure
    ann["well_formed"] = ""         # y / n
    ann["is_refusal"] = ""          # y / n
    ann["notes"] = ""
    ann.to_csv("results/human_annotation_set.csv", index=False)

    # a second copy of an overlap subset, for inter-annotator agreement
    ov = ann.head(args.overlap).copy()
    ov.to_csv("results/human_annotation_overlap.csv", index=False)

    # ---- key file, kept separate ----
    key = sample[["item_id", "dataset", "model_type", "variant", "similarity",
                  "nli_ok", "ent_fwd", "ent_bwd", "con_fwd", "ref_flipped",
                  "original_output", "rewrite_pred", "ref_orig", "ref_new",
                  "refusal", "stratum", "sampling_weight"]]
    key.to_csv("results/human_annotation_key.csv", index=False)

    print("\nwritten:")
    print("  results/human_annotation_set.csv      <- annotate this one")
    print(f"  results/human_annotation_overlap.csv  <- {args.overlap} items for a second annotator")
    print("  results/human_annotation_key.csv      <- signals, do not open before annotating")


if __name__ == "__main__":
    main()
