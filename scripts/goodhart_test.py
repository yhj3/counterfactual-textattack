"""Does optimizing the reported metric select worse prompts? (Goodhart test)

Naive ASR  = flip rate, what a red-teaming pipeline reports today.
Valid ASR  = flips that also (a) keep meaning, (b) are not reproduced by an
             independent referee, (c) are well-formed, (d) are not refusals.

If the two rank prompts the same way, the reported metric is a usable proxy.
If they rank them oppositely, optimizing the reported metric actively selects
the worse prompt -- Goodhart's law at the level of a single design choice.
"""
import re

import pandas as pd
from scipy.stats import spearmanr, fisher_exact

SIM_MIN = 0.75
REFUSAL = re.compile(r"^\s*(I cannot|I can.t|I.m sorry|I apologize|As an AI|I must decline|I will not)", re.I)

ref = pd.read_csv("results/referee_check.csv")
ref["flipped"] = ref.flipped.astype(bool)
ref["ref_flipped"] = ref.ref_flipped.astype(bool)
ref["refusal"] = ref.rewrite.astype(str).str.contains(REFUSAL)
ref["agree_orig"] = ref.ref_orig == ref.original_output

ref["naive"] = ref.flipped
ref["valid"] = (
    ref.flipped
    & (ref.similarity >= SIM_MIN)
    & (~ref.ref_flipped)
    & ref.agree_orig
    & (~ref.refusal)
)

cells = (
    ref.groupby(["dataset", "model_type", "variant"])
    .agg(n=("naive", "size"), naive=("naive", "mean"), valid=("valid", "mean"))
    .reset_index()
)
cells["cell"] = cells.dataset + "/" + cells.model_type
print("=== per (dataset x target x prompt) ===")
print(cells.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

rho, p = spearmanr(cells.naive, cells.valid)
print(f"\nSpearman(Naive ASR, Valid ASR) over {len(cells)} conditions: rho = {rho:.3f}  (p = {p:.3f})")

# the decision a practitioner actually makes: pick a prompt within each cell
print("\n=== the choice a practitioner makes: which prompt wins, by each metric ===")
rows = []
for cell, g in cells.groupby("cell"):
    sp = g[g.variant == "specific"].iloc[0]
    ge = g[g.variant == "generic"].iloc[0]
    naive_pick = "specific" if sp.naive > ge.naive else "generic"
    valid_pick = "specific" if sp.valid > ge.valid else ("generic" if ge.valid > sp.valid else "tie")
    rows.append({"cell": cell, "naive_specific": sp.naive, "naive_generic": ge.naive,
                 "valid_specific": sp.valid, "valid_generic": ge.valid,
                 "naive_picks": naive_pick, "valid_picks": valid_pick,
                 "reversed": naive_pick != valid_pick})
choice = pd.DataFrame(rows)
print(choice.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nreversed in {choice['reversed'].sum()} of {len(choice)} conditions")

print("\n=== pooled ===")
for v in ["specific", "generic"]:
    g = ref[ref.variant == v]
    print(f"  {v:9s} naive {g.naive.sum():3d}/{len(g)} = {g.naive.mean():.3f}   "
          f"valid {g.valid.sum():3d}/{len(g)} = {g.valid.mean():.3f}")
sp, ge = ref[ref.variant == "specific"], ref[ref.variant == "generic"]
a, b = sp.valid.sum(), len(sp) - sp.valid.sum()
c, d = ge.valid.sum(), len(ge) - ge.valid.sum()
print(f"  Fisher on Valid ASR: p = {fisher_exact([[a, b], [c, d]])[1]:.4f}")

cells.to_csv("results/goodhart_cells.csv", index=False)
choice.to_csv("results/goodhart_choice.csv", index=False)
print("\nwritten to results/goodhart_cells.csv and results/goodhart_choice.csv")
