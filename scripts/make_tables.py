"""Build the three result tables of the revised study from prompt_study.csv."""

import argparse

import numpy as np
import pandas as pd

FLIP_MIN, SIM_MIN, PPL_MAX = 0.6, 0.75, 150.0


def fmt(x, nd=3):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"


def table1(df):
    """Prompt specificity, dataset held fixed (single-sentence tasks)."""
    s = df[df.task == "single"]
    rows = []
    for (ds, mt), g in s.groupby(["dataset", "model_type"]):
        sp = g[g.variant == "specific"]
        ge = g[g.variant == "generic"]
        rows.append({
            "dataset": ds, "model": mt,
            "n": len(sp),
            "flip_specific": sp.flipped.mean() if len(sp) else np.nan,
            "flip_generic": ge.flipped.mean() if len(ge) else np.nan,
            "delta": (sp.flipped.mean() - ge.flipped.mean()) if len(sp) and len(ge) else np.nan,
            "sim_specific": sp.similarity.mean(), "sim_generic": ge.similarity.mean(),
            "ppl_specific": sp.perplexity.replace(np.inf, np.nan).median(),
            "ppl_generic": ge.perplexity.replace(np.inf, np.nan).median(),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "model"])


def table2(df):
    """Structure preservation on sentence-pair tasks."""
    p = df[df.task == "pair"]
    rows = []
    for (ds, mt), g in p.groupby(["dataset", "model_type"]):
        for variant in ["generic", "structured"]:
            v = g[g.variant == variant]
            if not len(v):
                continue
            valid = v[v.structure_ok]
            rows.append({
                "dataset": ds, "model": mt, "prompt": variant, "n": len(v),
                "structure_ok": v.structure_ok.mean(),
                "flip_among_valid": valid.flipped.mean() if len(valid) else np.nan,
                "flip_overall": v.flipped.mean(),
            })
    return pd.DataFrame(rows).sort_values(["dataset", "model", "prompt"])


def table3(df):
    """What the three-stage filter actually trades away."""
    s = df[df.task == "single"].copy()
    s["ppl_f"] = s.perplexity.replace(np.inf, np.nan)
    stages = [
        ("all rewrites", pd.Series(True, index=s.index)),
        ("+ flips the classifier", s.flipped),
        ("+ similarity >= 0.75", s.flipped & (s.similarity >= SIM_MIN)),
        ("+ perplexity <= 150", s.flipped & (s.similarity >= SIM_MIN) & (s.ppl_f <= PPL_MAX)),
    ]
    rows = []
    base = len(s)
    for name, mask in stages:
        sub = s[mask.fillna(False)]
        rows.append({
            "stage": name, "kept": len(sub), "frac_of_all": len(sub) / base if base else np.nan,
            "mean_similarity": sub.similarity.mean(),
            "median_perplexity": sub.ppl_f.median(),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df["flipped"] = df.flipped.astype(bool)

    t1, t2, t3 = table1(df), table2(df), table3(df)
    for name, t in [("table1_prompt", t1), ("table2_structure", t2), ("table3_filter", t3)]:
        print(f"\n================ {name} ================")
        print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        t.to_csv(f"{args.outdir}/{name}.csv", index=False)

    s = df[df.task == "single"]
    sp, ge = s[s.variant == "specific"], s[s.variant == "generic"]
    print("\n================ headline ================")
    print(f"single-sentence rewrites: n={len(s)}")
    print(f"  task-specific prompt : flip {sp.flipped.mean():.3f}  sim {sp.similarity.mean():.3f}"
          f"  ppl median {sp.perplexity.replace(np.inf,np.nan).median():.0f}")
    print(f"  generic prompt       : flip {ge.flipped.mean():.3f}  sim {ge.similarity.mean():.3f}"
          f"  ppl median {ge.perplexity.replace(np.inf,np.nan).median():.0f}")
    wins = (t1.delta > 0).sum()
    print(f"  specific > generic in {wins}/{len(t1)} dataset x model cells")

    p = df[df.task == "pair"]
    if len(p):
        pg = p[p.variant == "generic"]; ps = p[p.variant == "structured"]
        print(f"\nsentence-pair rewrites: n={len(p)}")
        print(f"  generic prompt   : structure kept {pg.structure_ok.mean():.3f}")
        print(f"  structured prompt: structure kept {ps.structure_ok.mean():.3f}")


if __name__ == "__main__":
    main()
