# Counterfactual Attacks on Text Classifiers

Code and results for *"Using Counterfactuals to Achieve TextAttack: Experiments
with TextFooler and LLaMA-2"* ([paper](paper/counterfactual-textattack.pdf)).

Standard attacks on text classifiers, such as TextFooler, swap individual words
for synonyms. This project asks what happens if a language model rewrites the
whole sentence instead — and adds a filter that keeps only rewrites that still
mean what the original meant. That filter is the point: an LLM rewrite is
evidence of a model weakness only if it stays faithful to the input. A rewrite
that quietly changes the meaning and then "fools" the classifier has not found
a vulnerability, it has just asked a different question.

Two findings came out of it: rewriting helps most on tasks that require
reasoning across sentences (RTE, QNLI) rather than on sentiment, where swapping
one word is already enough; and susceptibility grows with model size, with
RoBERTa-Large more vulnerable than BERT-Base.

## ⚠️ On the reference implementation

The original script implementing the three-stage semantic filter was lost.
[`src/semantic_filter.py`](src/semantic_filter.py) is a **reference
implementation written from Algorithm 1 of the paper**, not the script that was
actually run. **The numbers reported in the paper and in `results/` come from
the original experimental run, not from this file.**

The reconstruction follows the paper's thresholds exactly. Two places where the
paper's notation is ambiguous or self-inconsistent are marked with `NOTE` in the
source, together with the reading we adopted and why. Everything else in this
repository — the attack driver, the LLaMA-2 refinement, the comparison tooling,
and all result CSVs — is the original code and the original output.

## The filter

A generated candidate is kept only if all three conditions hold:

| Condition | Threshold | What it rules out |
|---|---|---|
| flip rate `φ(g)` | ≥ 0.6 | rewrites that do not actually change the decision |
| similarity `s(g)` | ≥ 0.75 | rewrites that drifted away from the original meaning |
| perplexity `p(g)` | ≤ 150 | rewrites that are not fluent English |

In the original run this discarded 37.7% of generated candidates while keeping
91% of the effective adversarial examples.

## Layout

```
src/semantic_filter.py   three-stage filter — REFERENCE IMPLEMENTATION (see above)
src/refiner.py           LLaMA-2 rewriting of an adversarial example
src/run_attack.py        end-to-end: TextFooler → LLaMA-2 rewrite → log
src/custom_attack.py     TextFooler with its semantic constraints removed
scripts/compare_results.py   builds the comparison CSVs in results/
scripts/check_env.py     CUDA / transformers sanity check
results/                 evaluation output from the original run
paper/                   the write-up
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...          # Llama-2 is gated on the Hugging Face Hub
export NLTK_DATA=...         # only if WordNet is not in the default location
```

A single 24 GB GPU is enough: Llama-2-7b-chat is loaded in 8-bit alongside the
victim classifier.

## Running

```bash
python scripts/check_env.py
CUDA_VISIBLE_DEVICES=0 ./scripts/run_attack_with_refinement.sh

# rebuild the comparison table from two result CSVs
python scripts/compare_results.py SUMMARY.csv REPORT.csv -o results/comparison_results.csv
```

## Results

`results/` holds the output of the original run:

| File | Contents |
|---|---|
| `comparison_results.csv` | attack success / perturbation / grammar, benchmark vs. recomputed |
| `comparison_improtved_results.csv` | the same comparison after the improved pipeline (filename typo is preserved from the original run) |
| `test.csv` (514 rows), `test_roberta_pwws_wnli.csv` (71 rows) | per-example attack records, including skipped and failed cases |
| `textfooler_results.csv`, `textfooler_results22.csv` | raw TextFooler output before rewriting |

## Attribution

This work builds on [TextAttack](https://github.com/QData/TextAttack) (QData,
MIT licensed), which provides the attack recipes, model wrappers, and dataset
loaders used here. TextAttack is a dependency rather than a copy: install it
from PyPI with the rest of `requirements.txt`. The attack recipe TextFooler is
due to Jin et al. (2020); LLaMA-2 is due to Touvron et al. (2023).

Everything under `src/`, `scripts/`, and `results/` is my own work, done as a
research assistant at FOCAL Lab, UIUC, with Prof.
[Gagandeep Singh](https://ggndpsngh.github.io/). Released with his approval.

## Contact

Yihang Jiao — yihangj3@illinois.edu · https://yhj3.github.io
