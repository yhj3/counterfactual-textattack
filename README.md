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

## ⚠️ On the reference implementation and what re-checking found

`src/semantic_filter.py` is a **reference implementation written from Algorithm 1
of the paper**. It is not the script that produced the paper's numbers.

While reconstructing it, the original pipeline was located (in the lab's
benchmark directory, not in this repository) and audited. Three things came out
of that audit, and they matter more than the reconstruction:

1. **The filter described in Algorithm 1 was never implemented.** The thresholds
   `s(g) >= 0.75` and `p(g) <= 150` appear nowhere in the pipeline. The `0.6`
   that Algorithm 1 attaches to the flip rate is, in the code, `min_cos_sim=0.6`
   — a word-embedding constraint passed to the *attack*, not a filter on
   generations.
2. **The classifier was never run on the rewrites.** LLaMA output was written
   into a `llama_text` column, and the reported attack success rate was computed
   from TextAttack's `result_type`, which describes the word-substitution attack
   that came *before* the rewrite. The rewrites did not enter the numbers.
3. **The paper's Table 1 compares different sample sizes.** Each "Flipped (%)"
   value is the attack's success rate over 20 examples; each "Normal (%)" value
   is the same attack over 277–500 examples. All five rows reproduce exactly
   from those two runs.

`scripts/reevaluate_llama_rewrites.py` supplies the missing evaluation: it runs
the victim classifier on the stored rewrites. Results are in
`results/corrected_table1.csv` and summarized below.

## Corrected evaluation

Rewrites exist only for examples the word-substitution attack had already
broken, so the question is how many of those attacks *survive* being rewritten.

| Task type | Configs | Result |
|---|---|---|
| Single-sentence (imdb, sst2, mr, ag_news, cola) | 20 | rewrites still flip the classifier **44.5%** of the time on average |
| Sentence-pair (rte, qnli, mrpc) | 12 | only **100 of 198** rewrites kept the two-segment structure and were valid classifier inputs at all |

Rewriting therefore loses more than half of the attacks it is applied to. The
spread across datasets is large and lines up with the prompts: `imdb` and
`ag_news`, the two datasets with hand-written prompt templates, retain 67–100%,
while `cola`, `mr`, and `sst2`, which fell through to the generic template,
retain 0–46%.

For RTE — the paper's headline result — **zero** of the 31 rewrites under
`bert_pwws_rte` and `roberta_pwws_rte` preserved the premise/hypothesis
structure, so none of them could be fed to an RTE classifier.

## The filter

A generated candidate is kept only if all three conditions hold:

| Condition | Threshold | What it rules out |
|---|---|---|
| flip rate `φ(g)` | ≥ 0.6 | rewrites that do not actually change the decision |
| similarity `s(g)` | ≥ 0.75 | rewrites that drifted away from the original meaning |
| perplexity `p(g)` | ≤ 150 | rewrites that are not fluent English |

The paper reports that this discarded 37.7% of candidates while keeping 91% of
the effective adversarial examples. Those figures could not be traced to any
code in the original pipeline; see the audit above.

## Layout

```
src/semantic_filter.py   three-stage filter — REFERENCE IMPLEMENTATION (see above)
src/refiner.py           LLaMA-2 rewriting of an adversarial example
src/run_attack.py        end-to-end: TextFooler → LLaMA-2 rewrite → log
src/custom_attack.py     TextFooler with its semantic constraints removed
scripts/reevaluate_llama_rewrites.py  runs the victim classifier on the stored rewrites
scripts/run_filter_pilot.py  applies the reconstructed filter to recorded attack results
scripts/run_llm_filter_experiment.py  generate rewrites with LLaMA-2, then filter
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
