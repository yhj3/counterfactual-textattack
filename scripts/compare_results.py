"""Compare TextAttack's own attack summary against our recomputed report.

Produces the comparison CSVs in `results/`: for each (model, attack, dataset)
cell it lines up attack success rate, average perturbation, and grammar score
from the two sources and reports the difference. Disagreements are the point —
they show where the benchmark's reported numbers and a recomputation diverge.

Usage:
    python scripts/compare_results.py SUMMARY_CSV REPORT_CSV [-o OUT_CSV]
"""

import argparse

import pandas as pd


def load_and_process_summary(filepath):
    """Load and normalize TextAttack's attack summary CSV."""
    df = pd.read_csv(filepath, dtype={"Model": str, "Attack": str, "Dataset": str})
    df.columns = df.columns.str.lower()

    # Percentages in the summary are 0-100; convert to decimals.
    df["asr (%)"] = df["asr (%)"] / 100
    df = df.rename(
        columns={
            "asr (%)": "asr",
            "avg perturbed %": "avg_perturb",
            "grammar score": "pert_grammar",
        }
    )

    df["model"] = df["model"].astype(str).str.title()
    df["attack"] = df["attack"].astype(str).str.title()
    df["dataset"] = df["dataset"].astype(str).str.lower()
    return df


def load_and_process_report(filepath):
    """Load our recomputed report and average any duplicate rows."""
    df = pd.read_csv(filepath, dtype={"model": str, "attack": str, "dataset": str})

    df["dataset"] = (
        df["dataset"].astype(str).str.replace(".csv", "", regex=False).str.lower()
    )
    df["model"] = df["model"].astype(str).str.title()
    df["attack"] = df["attack"].astype(str).str.title()

    agg_rules = {
        "asr": "mean",
        "avg_perturb": "mean",
        "orig_grammar": "mean",
        "pert_grammar": "mean",
    }
    return df.groupby(["model", "attack", "dataset"]).agg(agg_rules).reset_index()


def compare_metrics(summary_path, report_path):
    """Join the two sources and compute per-cell differences."""
    summary_df = load_and_process_summary(summary_path)
    report_df = load_and_process_report(report_path)

    merged = pd.merge(
        summary_df,
        report_df,
        on=["model", "attack", "dataset"],
        suffixes=("_summary", "_report"),
    )

    merged["asr_diff"] = merged["asr_summary"] - merged["asr_report"]
    merged["avg_perturb_diff"] = merged["avg_perturb_summary"] - merged["avg_perturb_report"]
    merged["grammar_diff"] = merged["pert_grammar_summary"] - merged["pert_grammar_report"]

    comparison = merged[
        [
            "model", "attack", "dataset",
            "asr_summary", "asr_report", "asr_diff",
            "avg_perturb_summary", "avg_perturb_report", "avg_perturb_diff",
            "pert_grammar_summary", "pert_grammar_report", "grammar_diff",
        ]
    ]
    return comparison.sort_values(by=["model", "attack", "dataset"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary_csv", help="TextAttack attack_summary.csv")
    ap.add_argument("report_csv", help="recomputed attack_report.csv")
    ap.add_argument("-o", "--out", default="comparison_results.csv")
    args = ap.parse_args()

    result = compare_metrics(args.summary_csv, args.report_csv)
    print("\nComparison Results:")
    print(result.round(3))
    result.to_csv(args.out, index=False)
    print(f"\nResults saved to {args.out}")
